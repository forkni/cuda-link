// spout_bridge.cpp — native CUDA <-> D3D11 <-> Spout bridge for cuda-link-spout.
//
// Implements the SpoutBackend surface (see _backend.py) as a pybind11 module named
// `_spout_bridge`. Windows-only: requires CUDA 12.x/13.x, D3D11, and the Spout2 SDK
// (spoutDX). This is the simple "2-copy" path of docs/competitive/spout-bridge-design.md
// (§5.1/§5.2): one CUDA de-swizzle copy (linear <-> CUDA array) + Spout's internal copy.
//
// Build: see spout/CMakeLists.txt. This file is intentionally not compiled in CI on
// Linux; the pure-Python layer is tested via FakeSpoutBackend. Validate on a Windows
// CUDA box before release.
//
// NOTE: treat as a reviewed reference implementation pending Windows compile/smoke test.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include "SpoutDX.h"  // Spout2 SDK (spoutDX class)

namespace py = pybind11;

namespace {

void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

// Create an ID3D11Device on the DXGI adapter whose LUID matches CUDA `device`.
// Pinning the adapter to the CUDA device is the #1 correctness requirement (§5.0).
ID3D11Device* createDeviceForCudaDevice(int device, ID3D11DeviceContext** outCtx) {
    cudaDeviceProp prop{};
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory))) {
        throw std::runtime_error("CreateDXGIFactory1 failed");
    }

    IDXGIAdapter1* chosen = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC desc{};
        adapter->GetDesc(&desc);
        // cudaDeviceProp.luid is an 8-byte LUID matching DXGI_ADAPTER_DESC.AdapterLuid.
        if (prop.luidDeviceNodeMask &&
            std::memcmp(&desc.AdapterLuid, prop.luid, sizeof(LUID)) == 0) {
            chosen = adapter;  // keep ref
            break;
        }
        adapter->Release();
    }
    factory->Release();
    if (!chosen) {
        throw std::runtime_error("No DXGI adapter matches the CUDA device LUID (multi-GPU mismatch?)");
    }

    ID3D11Device* dev = nullptr;
    D3D_FEATURE_LEVEL fl{};
    HRESULT hr = D3D11CreateDevice(chosen, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
                                   D3D11_SDK_VERSION, &dev, &fl, outCtx);
    chosen->Release();
    if (FAILED(hr)) throw std::runtime_error("D3D11CreateDevice failed");
    return dev;
}

struct Sender {
    spoutDX spout;
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    ID3D11Texture2D* tex = nullptr;           // shared texture we own + register with CUDA
    cudaGraphicsResource* cudaRes = nullptr;  // registration of `tex`
    int width = 0, height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    int device = 0;
};

struct Receiver {
    spoutDX spout;
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    ID3D11Texture2D* recvTex = nullptr;       // texture Spout hands us
    cudaGraphicsResource* cudaRes = nullptr;  // registration of recvTex
    void* dstBuf = nullptr;                   // linear device buffer (our copy target)
    size_t dstBytes = 0;
    int regW = 0, regH = 0;                   // geometry the current registration is valid for
    int device = 0;
};

std::mutex g_mu;
std::map<std::int64_t, Sender*> g_senders;
std::map<std::int64_t, Receiver*> g_receivers;
std::int64_t g_next = 0x6000'0000;

std::int64_t allocHandle() { return g_next += 0x10; }

}  // namespace

// --- sender -----------------------------------------------------------------

std::int64_t create_sender(const std::string& name, int width, int height, int dxgiFormat, int device) {
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");
    auto* s = new Sender();
    s->device = device;
    s->width = width;
    s->height = height;
    s->format = static_cast<DXGI_FORMAT>(dxgiFormat);
    s->dev = createDeviceForCudaDevice(device, &s->ctx);
    if (!s->spout.OpenDirectX11(s->dev)) {  // pin Spout to OUR (CUDA-matched) device
        delete s;
        throw std::runtime_error("spoutDX.OpenDirectX11 failed");
    }
    s->spout.SetSenderName(name.c_str());

    D3D11_TEXTURE2D_DESC td{};
    td.Width = width; td.Height = height; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = s->format; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    if (FAILED(s->dev->CreateTexture2D(&td, nullptr, &s->tex))) {
        delete s;
        throw std::runtime_error("CreateTexture2D (sender) failed");
    }
    cudaCheck(cudaGraphicsD3D11RegisterResource(&s->cudaRes, s->tex, cudaGraphicsRegisterFlagsNone),
              "cudaGraphicsD3D11RegisterResource (sender)");

    std::lock_guard<std::mutex> lk(g_mu);
    auto h = allocHandle();
    g_senders[h] = s;
    return h;
}

void send(std::int64_t handle, std::uintptr_t srcPtr, int srcPitch, int width, int height,
          int bytesPerPixel, std::uintptr_t stream) {
    Sender* s;
    { std::lock_guard<std::mutex> lk(g_mu); s = g_senders.at(handle); }
    auto cuStream = reinterpret_cast<cudaStream_t>(stream);

    cudaCheck(cudaGraphicsMapResources(1, &s->cudaRes, cuStream), "cudaGraphicsMapResources (send)");
    cudaArray_t arr = nullptr;
    cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, s->cudaRes, 0, 0),
              "cudaGraphicsSubResourceGetMappedArray (send)");
    cudaCheck(cudaMemcpy2DToArrayAsync(arr, 0, 0, reinterpret_cast<const void*>(srcPtr),
                                       srcPitch, static_cast<size_t>(width) * bytesPerPixel, height,
                                       cudaMemcpyDeviceToDevice, cuStream),
              "cudaMemcpy2DToArrayAsync (send)");
    cudaCheck(cudaGraphicsUnmapResources(1, &s->cudaRes, cuStream), "cudaGraphicsUnmapResources (send)");
    // Ensure the de-swizzle copy is complete before Spout reads `tex` (simple-path sync).
    cudaCheck(cudaStreamSynchronize(cuStream), "cudaStreamSynchronize (send)");
    if (!s->spout.SendTexture(s->tex)) {
        throw std::runtime_error("spoutDX.SendTexture failed");
    }
}

void close_sender(std::int64_t handle) {
    Sender* s = nullptr;
    { std::lock_guard<std::mutex> lk(g_mu); auto it = g_senders.find(handle);
      if (it != g_senders.end()) { s = it->second; g_senders.erase(it); } }
    if (!s) return;
    if (s->cudaRes) cudaGraphicsUnregisterResource(s->cudaRes);
    if (s->tex) s->tex->Release();
    s->spout.ReleaseSender();
    s->spout.CloseDirectX11();
    if (s->ctx) s->ctx->Release();
    if (s->dev) s->dev->Release();
    delete s;
}

// --- receiver ---------------------------------------------------------------

std::int64_t create_receiver(const std::string& name, int device) {
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");
    auto* r = new Receiver();
    r->device = device;
    r->dev = createDeviceForCudaDevice(device, &r->ctx);
    if (!r->spout.OpenDirectX11(r->dev)) {
        delete r;
        throw std::runtime_error("spoutDX.OpenDirectX11 (receiver) failed");
    }
    if (!name.empty()) r->spout.SetReceiverName(name.c_str());

    std::lock_guard<std::mutex> lk(g_mu);
    auto h = allocHandle();
    g_receivers[h] = r;
    return h;
}

// Returns (connected, new_frame, width, height, dxgi_format, dst_ptr).
py::tuple receive(std::int64_t handle, std::uintptr_t /*dstPtr*/, int /*dstPitch*/, std::size_t /*maxBytes*/) {
    Receiver* r;
    { std::lock_guard<std::mutex> lk(g_mu); r = g_receivers.at(handle); }

    if (!r->spout.ReceiveTexture(&r->recvTex) || r->recvTex == nullptr) {
        return py::make_tuple(false, false, 0, 0, 0, (std::uintptr_t)0);
    }
    const int w = (int)r->spout.GetSenderWidth();
    const int h = (int)r->spout.GetSenderHeight();
    const DXGI_FORMAT fmt = r->spout.GetSenderFormat();
    if (!r->spout.IsFrameNew()) {
        return py::make_tuple(true, false, w, h, (int)fmt, (std::uintptr_t)0);
    }

    // (Re)register on first frame or geometry change.
    if (r->cudaRes == nullptr || r->regW != w || r->regH != h) {
        if (r->cudaRes) { cudaGraphicsUnregisterResource(r->cudaRes); r->cudaRes = nullptr; }
        cudaCheck(cudaGraphicsD3D11RegisterResource(&r->cudaRes, r->recvTex, cudaGraphicsRegisterFlagsNone),
                  "cudaGraphicsD3D11RegisterResource (receiver)");
        r->regW = w; r->regH = h;
    }
    // Size the linear destination buffer to the sender's format.
    int bpp = 4;
    if (fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) bpp = 8;
    else if (fmt == DXGI_FORMAT_R32G32B32A32_FLOAT) bpp = 16;
    const size_t need = (size_t)w * h * bpp;
    if (need > r->dstBytes) {
        if (r->dstBuf) cudaFree(r->dstBuf);
        cudaCheck(cudaMalloc(&r->dstBuf, need), "cudaMalloc (receiver dst)");
        r->dstBytes = need;
    }

    cudaCheck(cudaGraphicsMapResources(1, &r->cudaRes, 0), "cudaGraphicsMapResources (recv)");
    cudaArray_t arr = nullptr;
    cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, r->cudaRes, 0, 0),
              "cudaGraphicsSubResourceGetMappedArray (recv)");
    cudaCheck(cudaMemcpy2DFromArray(r->dstBuf, (size_t)w * bpp, arr, 0, 0, (size_t)w * bpp, h,
                                    cudaMemcpyDeviceToDevice),
              "cudaMemcpy2DFromArray (recv)");
    cudaCheck(cudaGraphicsUnmapResources(1, &r->cudaRes, 0), "cudaGraphicsUnmapResources (recv)");
    cudaCheck(cudaStreamSynchronize(0), "cudaStreamSynchronize (recv)");

    return py::make_tuple(true, true, w, h, (int)fmt, (std::uintptr_t)r->dstBuf);
}

void close_receiver(std::int64_t handle) {
    Receiver* r = nullptr;
    { std::lock_guard<std::mutex> lk(g_mu); auto it = g_receivers.find(handle);
      if (it != g_receivers.end()) { r = it->second; g_receivers.erase(it); } }
    if (!r) return;
    if (r->cudaRes) cudaGraphicsUnregisterResource(r->cudaRes);
    if (r->dstBuf) cudaFree(r->dstBuf);
    r->spout.ReleaseReceiver();
    r->spout.CloseDirectX11();
    if (r->ctx) r->ctx->Release();
    if (r->dev) r->dev->Release();
    delete r;
}

std::int64_t adapter_luid(int device) {
    cudaDeviceProp prop{};
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::int64_t luid = 0;
    std::memcpy(&luid, prop.luid, sizeof(luid));
    return luid;
}

PYBIND11_MODULE(_spout_bridge, m) {
    m.doc() = "cuda-link-spout native CUDA<->D3D11<->Spout bridge";
    m.def("create_sender", &create_sender);
    m.def("send", &send);
    m.def("close_sender", &close_sender);
    m.def("create_receiver", &create_receiver);
    m.def("receive", &receive);
    m.def("close_receiver", &close_receiver);
    m.def("adapter_luid", &adapter_luid);
}
