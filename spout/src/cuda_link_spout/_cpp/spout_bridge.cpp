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
#include <cstring>  // std::memcmp / std::memcpy (LUID comparison + adapter_luid)
#include <map>
#include <memory>
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
    cudaGraphicsResource* cudaRes = nullptr;  // registration of regTex
    ID3D11Texture2D* regTex = nullptr;        // the exact texture cudaRes is bound to
    void* dstBuf = nullptr;                   // linear device buffer (our copy target)
    size_t dstBytes = 0;
    int regW = 0, regH = 0;                   // geometry the current registration is valid for
    int device = 0;
};

std::mutex g_mu;
// Values are shared_ptr (custom deleter = destroySender/destroyReceiver below) so a
// send/receive in flight after unlocking g_mu keeps the object alive even if
// close_sender/close_receiver erases the map entry concurrently on another thread —
// both send/receive and close_* run with the GIL released (see PYBIND11_MODULE) and
// would otherwise race a delete against an in-progress CUDA/Spout call.
std::map<std::int64_t, std::shared_ptr<Sender>> g_senders;
std::map<std::int64_t, std::shared_ptr<Receiver>> g_receivers;
std::int64_t g_next = 0x6000'0000;

std::int64_t allocHandle() { return g_next += 0x10; }

// Release everything a Sender owns and free it. Safe on partially-constructed
// senders (all fields null-checked), so it doubles as the error-path cleanup in
// create_sender — every early failure must funnel through here to avoid leaking
// the D3D11 device / Spout DX context / shared texture / CUDA registration.
void destroySender(Sender* s) {
    if (!s) return;
    if (s->cudaRes) cudaGraphicsUnregisterResource(s->cudaRes);
    if (s->tex) s->tex->Release();
    s->spout.ReleaseSender();
    s->spout.CloseDirectX11();
    if (s->ctx) s->ctx->Release();
    if (s->dev) s->dev->Release();
    delete s;
}

// Receiver counterpart of destroySender — also the error-path cleanup for
// create_receiver. Null-checks every field so it is safe on partial construction.
void destroyReceiver(Receiver* r) {
    if (!r) return;
    if (r->cudaRes) cudaGraphicsUnregisterResource(r->cudaRes);
    if (r->dstBuf) cudaFree(r->dstBuf);
    r->spout.ReleaseReceiver();
    r->spout.CloseDirectX11();
    if (r->ctx) r->ctx->Release();
    if (r->dev) r->dev->Release();
    delete r;
}

}  // namespace

// --- sender -----------------------------------------------------------------

std::int64_t create_sender(const std::string& name, int width, int height, int dxgiFormat, int device) {
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");
    auto* s = new Sender();
    s->device = device;
    s->width = width;
    s->height = height;
    s->format = static_cast<DXGI_FORMAT>(dxgiFormat);
    // Any failure past this point must release the partially-built Sender; route
    // every throw site through destroySender via a single catch.
    try {
        s->dev = createDeviceForCudaDevice(device, &s->ctx);
        if (!s->spout.OpenDirectX11(s->dev)) {  // pin Spout to OUR (CUDA-matched) device
            throw std::runtime_error("spoutDX.OpenDirectX11 failed");
        }
        s->spout.SetSenderName(name.c_str());

        D3D11_TEXTURE2D_DESC td{};
        td.Width = width; td.Height = height; td.MipLevels = 1; td.ArraySize = 1;
        td.Format = s->format; td.SampleDesc.Count = 1; td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        if (FAILED(s->dev->CreateTexture2D(&td, nullptr, &s->tex))) {
            throw std::runtime_error("CreateTexture2D (sender) failed");
        }
        cudaCheck(cudaGraphicsD3D11RegisterResource(&s->cudaRes, s->tex, cudaGraphicsRegisterFlagsNone),
                  "cudaGraphicsD3D11RegisterResource (sender)");
    } catch (...) {
        destroySender(s);
        throw;
    }

    std::lock_guard<std::mutex> lk(g_mu);
    auto h = allocHandle();
    g_senders[h] = std::shared_ptr<Sender>(s, destroySender);
    return h;
}

void spout_send(std::int64_t handle, std::uintptr_t srcPtr, int srcPitch, int width, int height,
                int bytesPerPixel, std::uintptr_t stream) {
    std::shared_ptr<Sender> s;
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_senders.find(handle);
      if (it == g_senders.end()) throw std::runtime_error("send: invalid or closed sender handle");
      s = it->second; }
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
    // Erasing the map entry drops our reference; the shared_ptr's deleter
    // (destroySender) only runs once every in-flight spout_send() also releases
    // its copy, so a concurrent send never touches a freed Sender.
    std::lock_guard<std::mutex> lk(g_mu);
    g_senders.erase(handle);
}

// --- receiver ---------------------------------------------------------------

std::int64_t create_receiver(const std::string& name, int device) {
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");
    auto* r = new Receiver();
    r->device = device;
    try {
        r->dev = createDeviceForCudaDevice(device, &r->ctx);
        if (!r->spout.OpenDirectX11(r->dev)) {
            throw std::runtime_error("spoutDX.OpenDirectX11 (receiver) failed");
        }
        if (!name.empty()) r->spout.SetReceiverName(name.c_str());
    } catch (...) {
        destroyReceiver(r);
        throw;
    }

    std::lock_guard<std::mutex> lk(g_mu);
    auto h = allocHandle();
    g_receivers[h] = std::shared_ptr<Receiver>(r, destroyReceiver);
    return h;
}

// Returns (connected, new_frame, width, height, dxgi_format, dst_ptr).
py::tuple receive(std::int64_t handle, std::uintptr_t /*dstPtr*/, int /*dstPitch*/, std::size_t /*maxBytes*/) {
    std::shared_ptr<Receiver> r;
    { std::lock_guard<std::mutex> lk(g_mu);
      auto it = g_receivers.find(handle);
      if (it == g_receivers.end()) throw std::runtime_error("receive: invalid or closed receiver handle");
      r = it->second; }

    // Use the no-argument ReceiveTexture() overload. The `ppTexture` overload requires
    // the caller to pre-allocate a receiver texture, handle IsUpdated(), and call
    // IsUpdated() to drain the m_bUpdated flag — a multi-step protocol that silently
    // returns NOT_CONNECTED on first connection when no texture is pre-allocated.
    // The no-arg overload handles connection establishment and update-flag lifecycle
    // internally; GetSenderTexture() then gives us m_pSharedTexture for the CUDA copy.
    if (!r->spout.ReceiveTexture()) {
        return py::make_tuple(false, false, 0, 0, 0, (std::uintptr_t)0);
    }
    const int w = (int)r->spout.GetSenderWidth();
    const int h = (int)r->spout.GetSenderHeight();
    const DXGI_FORMAT fmt = r->spout.GetSenderFormat();

    // IsUpdated() returns true on first connection and on sender geometry change,
    // AND resets the flag. Drain it here so subsequent calls flow through to GetNewFrame.
    // Return (connected=true, new_frame=false) so Python retries next poll.
    if (r->spout.IsUpdated()) {
        return py::make_tuple(true, false, w, h, (int)fmt, (std::uintptr_t)0);
    }
    if (!r->spout.IsFrameNew()) {
        return py::make_tuple(true, false, w, h, (int)fmt, (std::uintptr_t)0);
    }

    // GetSenderTexture() returns m_pSharedTexture — the sender's shared DX11 texture that
    // SpoutDX has opened on the receiver's D3D11 device. ReceiveTexture() already holds
    // (and releases) the frame mutex above, so we read the texture immediately after return
    // while the sender is not yet writing the next frame.
    ID3D11Texture2D* sharedTex = r->spout.GetSenderTexture();
    if (!sharedTex) {
        return py::make_tuple(true, false, w, h, (int)fmt, (std::uintptr_t)0);
    }

    // (Re)register on first frame, geometry change, OR when Spout hands back a
    // different texture object. spoutDX can recreate its receive texture at the
    // same dimensions (sender restart, a different same-size sender, internal
    // realloc); a registration bound to the old, now-released texture would read
    // freed memory. Keying on the texture pointer — not just geometry — catches that.
    if (r->cudaRes == nullptr || r->regTex != sharedTex || r->regW != w || r->regH != h) {
        if (r->cudaRes) { cudaGraphicsUnregisterResource(r->cudaRes); r->cudaRes = nullptr; }
        cudaCheck(cudaGraphicsD3D11RegisterResource(&r->cudaRes, sharedTex, cudaGraphicsRegisterFlagsNone),
                  "cudaGraphicsD3D11RegisterResource (receiver)");
        r->regTex = sharedTex; r->regW = w; r->regH = h;
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
    // Same shared_ptr handoff as close_sender: erase drops our reference, the
    // deleter runs once any in-flight receive() also releases its copy.
    std::lock_guard<std::mutex> lk(g_mu);
    g_receivers.erase(handle);
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
    // GIL is released for all functions that do not touch Python objects.
    // `receive` is excluded: it calls py::make_tuple at every return point and must
    // hold the GIL throughout.  A finer-grained py::gil_scoped_release block
    // inside receive() is left for a future pass once the code is compile-verified
    // on a Windows + CUDA + Spout2 box.
    using rgil = py::call_guard<py::gil_scoped_release>;
    m.def("create_sender",   &create_sender,   rgil{});
    m.def("send",            &spout_send,      rgil{});
    m.def("close_sender",    &close_sender,    rgil{});
    m.def("create_receiver", &create_receiver, rgil{});
    m.def("receive",         &receive);  // GIL held — see note above
    m.def("close_receiver",  &close_receiver,  rgil{});
    m.def("adapter_luid",    &adapter_luid,    rgil{});
}
