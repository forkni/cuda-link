"""Script TOP callbacks for the warning_emitter operator inside CUDAIPCLink.

Reads the status message written by RealTDHost (via ownerComp storage key
'cuda_link_status_msg') and re-emits it as a local addWarning badge on this
Script TOP. Produces a visible warning indicator inside the COMP alongside
the COMP-body tint set by RealTDHost.set_warning_status / set_error_status.

RealTDHost force-cooks this TOP on every status transition so the badge stays
in sync without relying on continuous cooking. Cook Type should be set to
'Off' (Pulse to Cook) in the TD parameter dialog.
"""


def onCook(scriptOp):
    msg = scriptOp.parent().fetch("cuda_link_status_msg", None)
    if msg:
        scriptOp.addWarning(str(msg))


def onSetupParameters(scriptOp):
    return


def onPulse(par):
    return
