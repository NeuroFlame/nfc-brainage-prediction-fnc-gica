import json
import os
import logging
import sys
import argparse

# ---------------------------------------------------------------------------
# Compatibility shim: cryptography < 3.5 does not expose pkcs12 as an
# attribute on the serialization namespace, causing nvflare 2.4.0's cert
# builder to fail with:
#   AttributeError: module '...serialization' has no attribute 'pkcs12'
#
# The pkcs12 submodule exists in all relevant versions but must be imported
# explicitly and attached before nvflare.lighter loads cert.py.
# ---------------------------------------------------------------------------
from cryptography.hazmat.primitives import serialization
if not hasattr(serialization, "pkcs12"):
    from cryptography.hazmat.primitives.serialization import pkcs12 as _pkcs12
    serialization.pkcs12 = _pkcs12

from provision.code.provision_run import provision_run

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)


def load_provision_input(provision_input_path: str) -> dict:
    try:
        with open(provision_input_path, 'r') as file:
            provision_input = json.load(file)
        logger.info(f"Provision input loaded from {provision_input_path}")
        return provision_input
    except Exception as e:
        logger.error(f"Failed to load provision input: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run provisioning script")
    parser.add_argument(
        '--input',
        default='/provisioning/provision_input.json',
        help='Path to the provision input file (default: /provisioning/provision_input.json)'
    )

    args = parser.parse_args()
    provision_input_path = args.input

    # Load provision input
    provision_input = load_provision_input(provision_input_path)
    path_run = os.path.join('/provisioning')

    # Extract arguments from provision input
    users = provision_input.get('users')  # list of {id, name}
    user_ids = [u['id'] for u in users]
    site_id_name_map = {u['id']: u['name'] for u in users}

    computation_parameters_dict = json.loads(provision_input.get('computation_parameters'))
    computation_parameters_dict['site_id_name_map'] = site_id_name_map

    # Derive owner_site from NeuroFLAME provisioning: the first user in the
    # users list is the computation owner (the site that initiated the run).
    # Injected automatically — not a user-facing parameter.
    owner_site_id = user_ids[0]
    owner_site_name = site_id_name_map.get(owner_site_id, owner_site_id)
    computation_parameters_dict['owner_site'] = owner_site_id
    logger.info(f"owner_site set to first provisioned user: '{owner_site_name}' (id: {owner_site_id})")

    computation_parameters = json.dumps(computation_parameters_dict)
    fed_learn_port = provision_input.get('fed_learn_port')
    admin_port = provision_input.get('admin_port')
    host_identifier = provision_input.get('host_identifier')

    print(f'user_ids: {user_ids}')
    print(f'path_run: {path_run}')
    print(f'computation_parameters: {computation_parameters}')
    print(f'fed_learn_port: {fed_learn_port}')
    print(f'admin_port: {admin_port}')
    print(f'host_identifier: {host_identifier}')

    # Call the provision_run function with the loaded arguments
    provision_run(
        user_ids=user_ids,
        path_run=path_run,
        path_app="/workspace/app",
        computation_parameters=computation_parameters,
        fed_learn_port=fed_learn_port,
        admin_port=admin_port,
        host_identifier=host_identifier,
    )


if __name__ == '__main__':
    main()
