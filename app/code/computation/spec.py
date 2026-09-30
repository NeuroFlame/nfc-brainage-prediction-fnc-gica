"""Declare the brain age FNC computation workflow."""

from framework import (
    ComputationSpec,
    local_step,
    remote_step,
    site_output_step,
    stepped_workflow,
)

from .inputs import load_inputs
from .local_math import prepare_site, train_member_model, train_owner_model
from .remote_math import designate_owner, finish_run, stack_member_weights
from .results import build_outputs

SPEC = ComputationSpec(
    workflow=stepped_workflow(
        local_step(fn=prepare_site, input_fn=load_inputs),
        remote_step(fn=designate_owner),
        local_step(fn=train_member_model),
        remote_step(fn=stack_member_weights),
        local_step(fn=train_owner_model),
        remote_step(fn=finish_run),
        site_output_step(fn=build_outputs),
    ),
)
