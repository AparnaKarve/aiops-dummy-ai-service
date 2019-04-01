"""Idle Cost Savings interface."""

from collections import defaultdict

import numpy as np
import pandas as pd


class AwsIdleCostSavingsResult:
    """Idle Cost Savings Result."""

    def __init__(self):
        """Initialize value that holds Result."""
        self.invalid_items = defaultdict(list)

    def add_recommendations(self, source_id, message):
        """Append conclusive Result message to Cluser Id.

        Only include clusters that need recommendation.
        """
        self.invalid_items[source_id].append(message)

    def to_dict(self):
        """Convert Idle Cost Savings Result instance to dict."""
        return self.invalid_items


class AwsIdleCostSavings:   #noqa  #Too few public methods
    """Idle Cost Savings."""

    # TODO:
    # 1. We shouldn't mark reserved instances for shutdown, we'll need data for
    # recognizing reserved instances and maybe the reservation timeline.
    # 2. We need inventory of autoscaling groups, since majority of nodes must
    # be terminated via autoscaling group (otherwise the deleted Vm will just
    # jump back)
    # 4. We should process the project resource quotas and recommend ideal
    # min&max for the autoscaling group
    # 5. We should correlate the measurements with utilization in time (cpu,
    # memory) and recommend to change the requests/limits?

    def __init__(self,
                 dataframes,
                 min_utilization=70.0,
                 max_utilization=80.0):
        """Initialize values required to run idle cost savings."""
        self.result = AwsIdleCostSavingsResult()
        self.container_nodes = dataframes.get('container_nodes')
        self.container_nodes_tags = dataframes.get('container_nodes_tags')
        self.containers = dataframes.get('containers')
        self.container_groups = dataframes.get('container_groups')
        self.container_projects = dataframes.get('container_projects')
        self.container_resource_quotas = dataframes.get(
            'container_resource_quotas'
        )
        self.flavors = dataframes.get('flavors')
        self.vms = dataframes.get('vms')
        self.sources = dataframes.get('sources')

        self.compute_roles = self.container_nodes_tags[
            self.container_nodes_tags['name'] ==
            "node-role.kubernetes.io/compute"
        ].copy()

        self.master_roles = self.container_nodes_tags[
            self.container_nodes_tags['name'] ==
            "node-role.kubernetes.io/master"
        ].copy()

        self.infra_roles = self.container_nodes_tags[
            self.container_nodes_tags['name'] ==
            "node-role.kubernetes.io/infra"
        ].copy()

        self.type_roles = self.container_nodes_tags[
            self.container_nodes_tags['name'] == "type"
        ].copy()

        self.instance_types = self.container_nodes_tags[
            self.container_nodes_tags['name'] ==
            "beta.kubernetes.io/instance-type"
        ].copy()

        self.min_utilization = min_utilization
        self.max_utilization = max_utilization

    def savings(self):
        """Get Savings for shutting down idle nodes."""
        # Load running containers
        active_containers = self._active_containers()
        # Load nodes having role compute
        compute_nodes = self._compute_nodes()
        # Group nodes by a cluster
        container_nodes_groups = compute_nodes[
            compute_nodes.lives_on_type == 'Vm'].groupby("source_id").groups

        for key in container_nodes_groups.keys():
            # Make recommendation for each cluster
            self._recommend_cost_savings(
                key,
                compute_nodes.loc[container_nodes_groups.get(key, [])],
                active_containers
            )

        return self.result

    def _get_quotas_scope(self, x_quotas_scope):
        scopes = x_quotas_scope.get('scopes', [])
        scope = None
        if 'NotTerminating' in scopes:
            scope = 'NotTerminating'
        return scope

    def _get_quotas_limit_cpu(self, x_quotas_limit_cpu):
        value = x_quotas_limit_cpu.get('hard', {}).get('limits.cpu')
        unit_cast_value = None
        if value:
            unit_cast_value = self._unit_cast(value)
        return unit_cast_value

    def _get_quotas_limit_memory(self, x_quotas_limit_memory):
        value = x_quotas_limit_memory.get('hard', {}).get('limits.memory')
        unit_cast_value = None
        if value:
            unit_cast_value = self._unit_cast(value)
        return unit_cast_value

    def _get_quotas_used_cpu(self, x_quotas_used_cpu):
        value = x_quotas_used_cpu.get('used', {}).get('limits.cpu')
        unit_cast_value = None
        if value:
            unit_cast_value = self._unit_cast(value)
        return unit_cast_value

    def _get_quotas_used_memory(self, x_quotas_used_memory):
        value = x_quotas_used_memory.get('used', {}).get('limits.memory')
        unit_cast_value = None
        if value:
            unit_cast_value = self._unit_cast(value)
        return unit_cast_value

    def _unit_cast(self, value):
        iec_60027_suffixes = {
            'Ki': 1,
            'Mi': 2,
            'Gi': 3,
            'Ti': 4,
            'Pi': 5,
            'Ei': 6,
            'Zi': 7,
            'Yi': 8
        }
        suffixes = {
            "d": "e-1",
            "c": "e-2",
            "m": "e-3",
            "μ": "e-6",
            "n": "e-9",
            "p": "e-12",
            "f": "e-15",
            "a": "e-18",
            "h": "e2",
            "k": "e3",
            "M": "e6",
            "G": "e9",
            "T": "e12",
            "P": "e15",
            "E": "e18"
        }

        unit_cast_value = float(value)

        if suffixes.get(value[-1]):
            unit_cast_value = \
                float("{}{}".format(value[0:-1], suffixes[value[-1]]))
        elif iec_60027_suffixes.get(value[-2:]):
            unit_cast_value = \
                float(value[0:-2]) * 1024**iec_60027_suffixes[value[-2:]]
        return unit_cast_value

    def _resource_quotas(self):
        if not np.any(self.container_resource_quotas):
            return self.container_resource_quotas

        # Load only active projects
        active_projects = self.container_projects[
            self.container_projects['status_phase'] == "Active"]
        # Load quotas of active projects
        resource_quotas = self.container_resource_quotas[
            self.container_resource_quotas[
                "container_project_id"
            ].astype('str').isin(active_projects['id'].astype('str'))
        ].copy()

        resource_quotas.loc[:, 'scope'] = resource_quotas.spec.apply(
            self._get_quotas_scope
        )

        resource_quotas.loc[:, 'limits_cpu'] = resource_quotas.status.apply(
            self._get_quotas_limit_cpu
        )
        resource_quotas.loc[:, 'limits_memory'] = resource_quotas.status.apply(
            self._get_quotas_limit_memory
        )
        resource_quotas.loc[:, 'used_cpu'] = resource_quotas.status.apply(
            self._get_quotas_used_cpu
        )
        resource_quotas.loc[:, 'used_memory'] = resource_quotas.status.apply(
            self._get_quotas_used_memory
        )

        return resource_quotas[resource_quotas['scope'] == "NotTerminating"]

    def _get_pod_uuid(self, pod_uuid):
        return self.container_groups[
            self.container_groups['id'] == pod_uuid]['source_ref'].item()

    def _get_container_node_id(self, container_node_id):
        return self.container_groups[
            self.container_groups['id'] == container_node_id
        ]['container_node_id'].item()

    def _get_project_name(self, project_name):
        return self.container_groups[
            self.container_groups['id'] == project_name
        ]['container_project_id'].item()

    def _active_containers(self):
        """Return containers that are active/running."""
        containers = self.containers

        containers.loc[:, 'pod_uuid'] = containers.container_group_id.apply(
            self._get_pod_uuid)
        containers.loc[
            :, 'container_node_id'] = containers.container_group_id.apply(
                self._get_container_node_id)
        containers.loc[
            :, 'project_name'] = containers.container_group_id.apply(
                self._get_project_name)
        containers.loc[:, 'cpu_limit_or_request'] = containers.id.apply(
            self._get_container_cpu_limit)
        containers.loc[:, 'memory_limit_or_request'] = containers.id.apply(
            self._get_container_memory_limit)

        return containers[containers['container_group_id'].isin(
            self.container_groups['id'])]

    def _get_container_cpu_limit(self, container_cpu_limit):
        container = self.containers[
            self.containers.id == container_cpu_limit
        ].iloc[0]

        container_cpu_limit = container.cpu_request
        if container.cpu_limit and not pd.isnull(container.cpu_limit):
            container_cpu_limit = container.cpu_limit
        return container_cpu_limit

    def _get_container_memory_limit(self, container_memory_limit):
        container = self.containers[
            self.containers.id == container_memory_limit
        ].iloc[0]

        container_memory_limit = container.memory_request
        if container.memory_limit and not pd.isnull(container.memory_limit):
            container_memory_limit = container.memory_limit
        return container_memory_limit

    def _get_host_inventory_uuid(self, x_host_inventory_uuid):
        vm = self.vms[self.vms['id'] == x_host_inventory_uuid]

        host_inventory_uuid = ""
        if np.any(vm):
            host_inventory_uuid = vm['host_inventory_uuid'].item()
        return host_inventory_uuid

    def _get_amount_of_pods(self, amount_of_pods):
        return len(self.container_groups[
            self.container_groups['container_node_id'] == amount_of_pods])

    def _compute_nodes(self):
        """Return nodes dataframe that have role compute.

        will also return nodes having multiple roles.
        """
        nodes = self.container_nodes.copy()
        nodes.loc[:, 'instance_type'] = nodes.id.apply(self._get_instance_type)
        nodes.loc[:, 'role'] = nodes.id.apply(self._get_type)
        nodes.loc[:, 'flavor_cpus'] = nodes.instance_type.apply(
            self._get_flavor_cpu)
        nodes.loc[:, 'flavor_memory'] = nodes.instance_type.apply(
            self._get_flavor_memory)

        nodes.loc[:, '#pods'] = nodes.id.apply(
            self._get_amount_of_pods)

        nodes.loc[:, 'host_inventory_uuid'] = nodes.lives_on_id.apply(
            self._get_host_inventory_uuid)

        compute_container_nodes = nodes[nodes["role"].str.contains("compute")]
        return compute_container_nodes

    def _get_instance_type(self, x_instance_type):
        instance_type = self.instance_types[
            self.instance_types["container_node_id"] == x_instance_type]

        instance_type_value = ""
        if np.any(instance_type) and instance_type['value'].item():
            instance_type_value = instance_type['value'].item()
        else:
            vm_id = self.container_nodes[
                self.container_nodes['id'] == x_instance_type
            ]

            if np.any(vm_id) or np.any(self.vms):
                vm = self.vms[self.vms["id"] == vm_id['lives_on_id'].item()]
                if np.any(vm):
                    flavor = self.instance_types[
                        self.instance_types["id"] == vm["flavor_id"].item()]
                    if np.any(flavor):
                        instance_type_value = flavor["source_ref"].item()

        return instance_type_value

    def _get_type(self, x_type):
        special_type = self.type_roles[
            self.type_roles["container_node_id"] == x_type]
        if np.any(special_type):
            return special_type['value'].item()

        compute_role = self.compute_roles[
            self.compute_roles["container_node_id"] == x_type]
        master_role = self.master_roles[
            self.master_roles["container_node_id"] == x_type]
        infra_role = self.infra_roles[
            self.infra_roles["container_node_id"] == x_type]

        roles = []
        if np.any(compute_role):
            roles.append("compute")
        if np.any(infra_role):
            roles.append("infra")
        if np.any(master_role):
            roles.append("master")

        return ",".join(roles)

    def _get_flavor_cpu(self, x_flavor_cpu):
        if not np.any(self.flavors):
            return None

        cpu_count = self.flavors[
            self.flavors['cpus'].notnull() &
            (self.flavors['name'] == x_flavor_cpu)
        ]

        flavor_cpu = None
        if np.any(cpu_count):
            flavor_cpu = cpu_count.iloc[0]['cpus'].item()
        return flavor_cpu

    def _get_flavor_memory(self, x_flavor_memory):
        if not np.any(self.flavors):
            return None

        memory = self.flavors[
            self.flavors['memory'].notnull() &
            (self.flavors['name'] == x_flavor_memory)
        ]

        flavor_memory = None
        if np.any(memory):
            flavor_memory = memory.iloc[0]['memory'].item()
        return flavor_memory

    def _cpu_utilization(self, container_nodes_group, containers):
        available = container_nodes_group['allocatable_cpus'].sum(
            axis=0,
            skipna=True
        )
        # TODO use request_or_limit, in a case that has only limit defined
        consumed = containers['cpu_request'].sum(
            axis=0,
            skipna=True
        )

        cpu_utilization = 0
        if available > 0:
            cpu_utilization = 100.0/available*consumed
        return cpu_utilization

    def _memory_utilization(self, container_nodes_group, containers):
        available = container_nodes_group['allocatable_memory'].sum(
            axis=0,
            skipna=True
        )
        consumed = containers['memory_limit_or_request'].sum(
            axis=0,
            skipna=True
        )

        memory_utilization = 0
        if available > 0:
            memory_utilization = 100.0/available*consumed
        return memory_utilization

    def _pods_utilization(self, remaining_nodes, all_nodes):
        available = remaining_nodes['allocatable_pods'].sum(
            axis=0,
            skipna=True
        )
        consumed = all_nodes['#pods'].sum(
            axis=0,
            skipna=True
        )

        pods_utilization = 0
        if available > 0:
            pods_utilization = 100.0/available*consumed
        return pods_utilization

    def _utilization(self, remaining_nodes, all_nodes, containers):
        cpu_utilization = self._cpu_utilization(
            remaining_nodes,
            containers
        )
        memory_utilization = self._memory_utilization(
            remaining_nodes,
            containers
        )
        pods_utilization = self._pods_utilization(
            remaining_nodes,
            all_nodes
        )

        return max([cpu_utilization, memory_utilization, pods_utilization])

    def _format_node_list(self, nodes):
        return nodes.loc[:, [
            "id",
            "host_inventory_uuid",
            "allocatable_memory",
            "allocatable_cpus",
            "allocatable_pods",
            "#pods"
        ]].sort_values(
            by='#pods',
            ascending=True
        )

    def _store_recommendation(
            self,
            source_id,
            all_nodes,
            remaining_nodes,
            containers
    ):
        cpu_utilization = self._cpu_utilization(all_nodes, containers)
        memory_utilization = self._memory_utilization(all_nodes, containers)
        pods_utilization = self._pods_utilization(all_nodes, all_nodes)

        optimized_cpu_utilization = self._cpu_utilization(
            remaining_nodes,
            containers
        )
        optimized_memory_utilization = self._memory_utilization(
            remaining_nodes,
            containers
        )
        optimized_pods_utilization = self._pods_utilization(
            remaining_nodes,
            all_nodes
        )

        shut_off_nodes = all_nodes[
            ~all_nodes['id'].isin(remaining_nodes['id'])]

        message = {}
        message['cluster_name'] = self.sources[(self.sources['id'] == source_id)].name.values[0]
        message['message'] = "For saving cost we can scale down nodes in a cluster"
        message['current_state'] = {
                "cpu_utilization": cpu_utilization,
                "memory_utilization": memory_utilization,
                "pods_utilization": pods_utilization,
                "nodes": self._format_node_list(all_nodes).to_dict('records')
        }
        message['after_scaledown'] = {
                "cpu_utilization": optimized_cpu_utilization,
                "memory_utilization": optimized_memory_utilization,
                "pods_utilization": optimized_pods_utilization,
                "nodes":
                    self._format_node_list(remaining_nodes).to_dict('records')
        }
        message['recommended_nodes_for_shut_down'] = \
            self._format_node_list(shut_off_nodes).to_dict('records')

        self.result.add_recommendations(source_id, message)

    def _recommend_cost_savings(
            self,
            source_id,
            container_nodes_group,
            active_containers
    ):
        """Recommend nodes that can be shutoff."""
        containers = \
            active_containers[
                active_containers["container_node_id"].isin(
                    container_nodes_group['id']
                )
            ]

        if self._utilization(container_nodes_group, container_nodes_group,
                             containers) >= self.min_utilization:
            return

        shut_off_nodes = []

        for _index, node in container_nodes_group.sort_values(
                by='#pods',
                ascending=True
        ).iterrows():

            # Look at the utilization if we shut off one more node
            shut_off_nodes.append(node.id)
            nodes = container_nodes_group[
                ~container_nodes_group['id'].isin(shut_off_nodes)
            ]
            utilization = self._utilization(
                nodes,
                container_nodes_group,
                containers
            )

            if utilization >= self.max_utilization:
                # We've removed too much nodes, util is over 100% now, lets
                # put back the last remove node and we should be in ideal state
                del shut_off_nodes[-1]
                nodes = container_nodes_group[
                    ~container_nodes_group['id'].isin(shut_off_nodes)
                ]

                self._store_recommendation(
                    source_id,
                    container_nodes_group,
                    nodes,
                    containers
                )

                return

            if len(nodes) <= 1:
                # We have only last node left, lets just keep that
                self._store_recommendation(
                    source_id,
                    container_nodes_group,
                    nodes,
                    containers
                )
                return

            if utilization >= self.min_utilization:
                # Utilization is over what we've specified, lets recommend
                # this state.
                self._store_recommendation(
                    source_id,
                    container_nodes_group,
                    nodes,
                    containers
                )
                return
