"""AWS Volume Type Validation interface."""
import logging
import sys
import numpy as np
import pandas as pd

from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.handlers = []
logger.addHandler(handler)


class AwsIdleCostSavingResult:
    """Validation Result Processing."""

    # def __init__(self):
    #     """Initialize value that holds Result."""
    #     self.volume_type_results = defaultdict(defaultdict)
    #     self.invalid_items = defaultdict(list)
    #     self.hosts = defaultdict(list)
    #
    # def add_recommendations(self, source_id, message):
    #     """Append conclusive Result message to Cluser Id. Only include wrong clusters"""
    #     self.invalid_items[source_id].append(message)
    #
    # def set_hosts(self, hosts):
    #     """Assign per-host with recommendations dict."""
    #     self.hosts = hosts
    #
    # def to_dict(self):
    #     """Convert Volume Type Results instance to dict."""
    #     self.volume_type_results['clusters'] = self.invalid_items
    #     self.volume_type_results['hosts'] = self.hosts
    #     return self.volume_type_results
    def __init__(self):
        self.invalid_items = defaultdict(list)

    def add(self, source_id, message):
        self.invalid_items[source_id].append(message)

    def to_dict(self):
        return self.invalid_items

    # def pprint(self):
    #     print(pprint.PrettyPrinter(indent=1).pprint(self.invalid_items))


class AwsIdleCostSavings:
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
        self.result = AwsIdleCostSavingResult()
        self.container_nodes = dataframes.get('container_nodes')
        self.container_nodes_tags = dataframes.get('container_nodes_tags')
        self.containers = dataframes.get('containers')
        self.container_groups = dataframes.get('container_groups')
        self.container_projects = dataframes.get('container_projects')
        self.container_resource_quotas = dataframes.get('container_resource_quotas')
        self.flavors = dataframes.get('flavors')
        self.vms = dataframes.get('vms')
        self.min_utilization = min_utilization
        self.max_utilization = max_utilization
        self.logger = logger
        if not self.logger:
            self.logger = logging.getLogger(__name__)

    def savings(self):
        # Load running containers
        active_containers = self._active_containers()
        # Load nodes having role compute
        compute_nodes = self._compute_nodes()
        # Load resource quotas
        resource_quotas = self._resource_quotas()
        # Group nodes by a cluster
        container_nodes_groups = compute_nodes[
            compute_nodes.lives_on_type == 'Vm'].groupby("source_id").groups
        self.logger.debug("")
        self.logger.debug("--- container_nodes_groups ---")
        self.logger.debug(container_nodes_groups)

        for key in container_nodes_groups.keys():
            # Make recommendation for each cluster
            self._recommend_cost_savings(
                key,
                compute_nodes.loc[container_nodes_groups.get(key, [])],
                active_containers,
                resource_quotas)

        return self.result

    def _get_quotas_scope(self, x):
        # x.get('hard', {}).get('scopes')
        scopes = x.get('scopes', [])
        for scope in scopes:
            if scope == "NotTerminating":
                return "NotTerminating"

    def _get_quotas_limit_cpu(self, x):
        value = x.get('hard', {}).get('limits.cpu')
        if value:
            return self._unit_cast(value)

    def _get_quotas_limit_memory(self, x):
        value = x.get('hard', {}).get('limits.memory')
        if value:
            return self._unit_cast(value)

    def _get_quotas_used_cpu(self, x):
        value = x.get('used', {}).get('limits.cpu')
        if value:
            return self._unit_cast(value)

    def _get_quotas_used_memory(self, x):
        value = x.get('used', {}).get('limits.memory')
        if value:
            return self._unit_cast(value)

    def _unit_cast(self, x):
        iec_60027_suffixes = {
            'Ki': 1, 'Mi': 2, 'Gi': 3, 'Ti': 4, 'Pi': 5, 'Ei': 6, 'Zi': 7,
            'Yi': 8
        }
        suffixes = {
            "d": "e-1", "c": "e-2", "m": "e-3", "μ": "e-6", "n": "e-9",
            "p": "e-12", "f": "e-15", "a": "e-18", "h": "e2", "k": "e3",
            "M": "e6", "G": "e9", "T": "e12", "P": "e15", "E": "e18"
        }

        if suffixes.get(x[-1]):
            return float("{}{}".format(x[0:-1], suffixes[x[-1]]))
        elif iec_60027_suffixes.get(x[-2:]):
            return float(x[0:-2]) * 1024**iec_60027_suffixes[x[-2:]]
        else:
            return float(x)

    def _resource_quotas(self):
        if not np.any(self.container_resource_quotas):
            return self.container_resource_quotas

        # Load only active projects
        active_projects = self.container_projects[
            self.container_projects['status_phase'] == "Active"]
        # Load quotas of active projects
        resource_quotas = self.container_resource_quotas[
            self.container_resource_quotas[
                "container_project_id"].astype('str').isin(
                    active_projects['id'].astype('str')
            )
        ].copy()

        resource_quotas.loc[:, 'scope'] = resource_quotas.spec.apply(
            self._get_quotas_scope)

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

    def _get_pod_uuid(self, x):
        return self.container_groups[
            self.container_groups['id'] == x]['source_ref'].item()

    def _get_container_node_id(self, x):
        return self.container_groups[
            self.container_groups['id'] == x]['container_node_id'].item()

    def _get_project_name(self, x):
        return self.container_groups[
            self.container_groups['id'] == x]['container_project_id'].item()

    def _active_containers(self):
        """
        Returns containers that are active/running.
        """
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

    def _get_container_cpu_limit(self, x):
        container = self.containers[self.containers.id == x].iloc[0]
        if container.cpu_limit and not pd.isnull(container.cpu_limit):
            return container.cpu_limit
        else:
            return container.cpu_request

    def _get_container_memory_limit(self, x):
        container = self.containers[self.containers.id == x].iloc[0]
        if container.memory_limit and not pd.isnull(container.memory_limit):
            return container.memory_limit
        else:
            return container.memory_request

    def _get_host_inventory_uuid(self, x):
        vm = self.vms[self.vms['id'] == x]
        if np.any(vm):
            return vm['host_inventory_uuid'].item()

    def _get_amount_of_pods(self, x):
        return len(self.container_groups[
            self.container_groups['container_node_id'] == x])

    def _compute_nodes(self):
        """
        Return nodes dataframe that have role compute, will return also nodes
        having multiple roles
        """
        tags = self.container_nodes_tags

        self.compute_roles = tags[
            tags['name'] == "node-role.kubernetes.io/compute"].copy()
        self.master_roles = tags[
            tags['name'] == "node-role.kubernetes.io/master"].copy()
        self.infra_roles = tags[
            tags['name'] == "node-role.kubernetes.io/infra"].copy()
        self.type_roles = tags[tags['name'] == "type"].copy()
        self.instance_types = tags[
            tags['name'] == "beta.kubernetes.io/instance-type"].copy()

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

        # func = lambda x: self.container_projects[
        # self.container_projects['id'].isin(
        # container_groups[
        # container_groups['container_node_id']==x]['container_project_id']
        # )].name.values
        # nodes.loc[:, 'projects'] = nodes.id.apply(func)

        self.logger.debug("=== Nodes: ===")
        self.logger.debug(nodes.reindex(columns=[
            "id", "source_id", "source_ref", "name", "memory",
            "flavor_memory", "cpus", "flavor_cpus",
            "instance_type", "role", "host_inventory_uuid",
            "#pods"]).sort_values(
            by='instance_type', ascending=False))

        compute_container_nodes = nodes[nodes["role"].str.contains("compute")]
        return compute_container_nodes

    def _get_instance_type(self, x):
        instance_type = self.instance_types[
            self.instance_types["container_node_id"] == x]
        if np.any(instance_type):
            if instance_type['value'].item():
                return instance_type['value'].item()
            else:
                return ""
        else:
            vm_id = self.container_nodes[
               self.container_nodes['id'] == x]

            if not np.any(vm_id) or not np.any(self.vms):
                return ""
            vm = self.vms[self.vms["id"] == vm_id['lives_on_id'].item()]
            if not np.any(vm):
                return ""
            flavor = self.instance_types[
                self.instance_types["id"] == vm["flavor_id"].item()]
            if np.any(flavor):
                return flavor["source_ref"].item()
            else:
                return ""

    def _get_type(self, x):
        special_type = self.type_roles[
            self.type_roles["container_node_id"] == x]
        if np.any(special_type):
            return special_type['value'].item()

        compute_role = self.compute_roles[
            self.compute_roles["container_node_id"] == x]
        master_role = self.master_roles[
            self.master_roles["container_node_id"] == x]
        infra_role = self.infra_roles[
            self.infra_roles["container_node_id"] == x]

        roles = []
        if np.any(compute_role):
            roles.append("compute")
        if np.any(infra_role):
            roles.append("infra")
        if np.any(master_role):
            roles.append("master")

        return ",".join(roles)

    def _get_flavor_cpu(self, x):
        if not np.any(self.flavors):
            return None

        cpu_count = self.flavors[
            self.flavors['cpus'].notnull() & (self.flavors['name'] == x)]
        if np.any(cpu_count):
            return cpu_count.iloc[0]['cpus'].item() #due to csv string conversion?
            # return cpu_count.iloc[0]['cpus']

    def _get_flavor_memory(self, x):
        if not np.any(self.flavors):
            return None

        memory = self.flavors[
            self.flavors['memory'].notnull() & (self.flavors['name'] == x)]
        if np.any(memory):
            return memory.iloc[0]['memory'].item() #//due to csv string conversion?
            # return memory.iloc[0]['memory']

    def _cluster_status(self, source_id, container_nodes_group, containers,
                        resource_quotas):
        self.logger.debug("==================================================")
        self.logger.debug("========= Cluster status for source id {} "
                          "=========".format(source_id))

        sum_of_mem_requests = containers['memory_request'].sum(
            axis=0, skipna=True)
        self.logger.debug("Sum of sum_of_mem_requests: {} GB".format(
            sum_of_mem_requests/1024**3))

        sum_of_cpu_requests = containers['cpu_request'].sum(
            axis=0, skipna=True)
        self.logger.debug("Sum of sum_of_cpu_requests: {} cores".format(
            sum_of_cpu_requests))

        sum_of_mem_limits = containers['memory_limit'].sum(
            axis=0, skipna=True)
        self.logger.debug("Sum of sum_of_mem_limits: {} GB".format(
            sum_of_mem_limits/1024**3))

        sum_of_cpu_limits = containers['cpu_limit'].sum(
            axis=0, skipna=True)
        self.logger.debug("Sum of sum_of_cpu_limits: {} cores".format(
            sum_of_cpu_limits))

        sum_of_mem_limits = containers['memory_limit_or_request'].sum(
            axis=0, skipna=True)
        self.logger.debug(
            "Sum of sum_of_memory_limits_or_request: {} GB".format(
                sum_of_mem_limits/1024**3))

        sum_of_cpu_limits = containers['cpu_limit_or_request'].sum(
            axis=0, skipna=True)
        self.logger.debug(
            "Sum of sum_of_cpu_limits_or_request: {} cores".format(
                sum_of_cpu_limits))

        sum_of_am = container_nodes_group['allocatable_memory'].sum(
            axis=0, skipna=True)
        self.logger.debug(
            "Sum of sum_of_available_memory by nodes: {} GiB".format(
                sum_of_am/1024**3))

        sum_of_am = container_nodes_group['flavor_memory'].sum(
            axis=0, skipna=True)
        self.logger.debug(
            "Sum of sum_of_available_memory by flavors: {} GiB".format(
                sum_of_am/1024**3))

        sum_of_ac = container_nodes_group['allocatable_cpus'].sum(
            axis=0, skipna=True)
        self.logger.debug(
            "Sum of sum_of_available_cpus by nodes: {} cores".format(
                sum_of_ac))

        sum_of_ac = container_nodes_group['flavor_cpus'].sum(
            axis=0, skipna=True)
        self.logger.debug(
            "Sum of sum_of_available_cpus by flavors: {} cores".format(
                sum_of_ac))

        sum_of_qlc = resource_quotas['limits_cpu'].sum(
            axis=0, skipna=True)
        self.logger.debug("Sum of quotas limits.cpu: {} cores".format(
            sum_of_qlc))

        sum_of_qlm = resource_quotas['limits_memory'].sum(
            axis=0, skipna=True)
        self.logger.debug("Sum of quotas limits.memory: {} GiB".format(
            sum_of_qlm/1024**3))

        # TODO show how many containers limits are consumed by containers
        # in a project without quotas
        sum_of_quc = resource_quotas['used_cpu'].sum(
            axis=0, skipna=True)
        self.logger.debug("Sum of quotas used.cpu: {} cores".format(
            sum_of_quc))

        sum_of_qum = resource_quotas['used_memory'].sum(
            axis=0, skipna=True)
        self.logger.debug("Sum of quotas used.memory: {} GiB".format(
            sum_of_qum/1024**3))

    def _cpu_utilization(self, container_nodes_group, containers):
        available = container_nodes_group['allocatable_cpus'].sum(
            axis=0, skipna=True)
        # TODO use request_or_limit, in a case that only limit is defined
        consumed = containers['cpu_request'].sum(
            axis=0, skipna=True)

        if available > 0:
            return 100.0/available*consumed

    def _memory_utilization(self, container_nodes_group, containers):
        available = container_nodes_group['allocatable_memory'].sum(
            axis=0, skipna=True)
        consumed = containers['memory_limit_or_request'].sum(
            axis=0, skipna=True)

        if available > 0:
            return 100.0/available*consumed

    def _pods_utilization(self, remaining_nodes, all_nodes):
        available = remaining_nodes['allocatable_pods'].sum(
            axis=0, skipna=True)
        consumed = all_nodes['#pods'].sum(
            axis=0, skipna=True)

        if available > 0:
            return 100.0/available*consumed

    def _utilization(self, remaining_nodes, all_nodes, containers):
        cpu_utilization = self._cpu_utilization(
            remaining_nodes, containers)
        memory_utilization = self._memory_utilization(
            remaining_nodes, containers)
        pods_utilization = self._pods_utilization(
            remaining_nodes, all_nodes
        )

        return max([cpu_utilization, memory_utilization, pods_utilization])

    def _format_node_list(self, nodes):
        return nodes.loc[:, [
            "id", "host_inventory_uuid", "allocatable_memory",
            "allocatable_cpus", "allocatable_pods", "#pods"]].sort_values(
            by='#pods', ascending=True)

    def _store_recommendation(self, source_id, all_nodes, remaining_nodes,
                              containers):
        cpu_utilization = self._cpu_utilization(all_nodes, containers)
        memory_utilization = self._memory_utilization(all_nodes, containers)
        pods_utilization = self._pods_utilization(all_nodes, all_nodes)

        optimized_cpu_utilization = self._cpu_utilization(
            remaining_nodes, containers)
        optimized_memory_utilization = self._memory_utilization(
            remaining_nodes, containers)
        optimized_pods_utilization = self._pods_utilization(
            remaining_nodes, all_nodes)

        shut_off_nodes = all_nodes[
            ~all_nodes['id'].isin(remaining_nodes['id'])]

        self.logger.debug("\n---- Optimized Utilization -----")
        self.logger.debug("Optimized CPU utilization {:.5}%".format(
            optimized_cpu_utilization))
        self.logger.debug("Optimized Memory utilization {:.5}%".format(
            optimized_memory_utilization))
        self.logger.debug("Optimized Pods utilization {:.5}%".format(
            optimized_pods_utilization))

        self.logger.debug("\n----- Remaining nodes after scale down sorted "
                          "by # of containers ---- ")
        self.logger.debug(self._format_node_list(remaining_nodes))

        message = {
            "message": "For saving cost we can scale down nodes in a cluster",
            "current_state": {
                "cpu_utilization": cpu_utilization,
                "memory_utilization": memory_utilization,
                "pods_utilization": pods_utilization,
                "nodes": self._format_node_list(
                    all_nodes).to_dict('records')
            },
            "after_scaledown": {
                "cpu_utilization": optimized_cpu_utilization,
                "memory_utilization": optimized_memory_utilization,
                "pods_utilization": optimized_pods_utilization,
                "nodes": self._format_node_list(
                    remaining_nodes).to_dict('records')
            },
            "recommended_nodes_for_shut_down": self._format_node_list(
                shut_off_nodes).to_dict('records'),
        }
        self.result.add(source_id, message)

    def _recommend_cost_savings(self, source_id, container_nodes_group,
                                active_containers, resource_quotas):
        """
        Recommend nodes that can be shutoff

        """
        containers = active_containers[
            active_containers["container_node_id"].isin(
                container_nodes_group['id'])]

        quotas = resource_quotas[resource_quotas['source_id'] == source_id]

        self._cluster_status(source_id, container_nodes_group, containers,
                             quotas)

        self.logger.debug("\n---- Utilization -----")
        self.logger.debug("CPU utilization {:.5}%".format(
            self._cpu_utilization(container_nodes_group, containers)))
        self.logger.debug("Memory utilization {:.5}%".format(
            self._memory_utilization(container_nodes_group, containers)))
        self.logger.debug("Pods utilization {:.5}%".format(
            self._pods_utilization(container_nodes_group,
                                   container_nodes_group)))

        self.logger.debug("\n----- Nodes sorted by # of containers ---- ")
        self.logger.debug(self._format_node_list(container_nodes_group))

        if self._utilization(container_nodes_group, container_nodes_group,
                             containers) < self.min_utilization:
            self.logger.debug(
                "Utilization of memory or cpu doesn't reach {}%, "
                "we can scale down nodes!".format(self.min_utilization))
        else:
            self.logger.debug("Utilization is ideal!")
            return

        shut_off_nodes = []

        for index, node in container_nodes_group.sort_values(
                by='#pods', ascending=True).iterrows():

            # Look at the utilization if we shut off one more node
            shut_off_nodes.append(node.id)
            nodes = container_nodes_group[
                ~container_nodes_group['id'].isin(shut_off_nodes)]
            utilization = self._utilization(
                nodes, container_nodes_group, containers)

            if utilization >= self.max_utilization:
                # We've removed too much nodes, util is over 100% now, lets
                # put back the last remove node and we should be in ideal state
                del shut_off_nodes[-1]
                nodes = container_nodes_group[
                    ~container_nodes_group['id'].isin(shut_off_nodes)]

                self._store_recommendation(source_id, container_nodes_group,
                                           nodes, containers)

                return

            elif len(nodes) <= 1:
                # We have only last node left, lets just keep that
                self._store_recommendation(source_id, container_nodes_group,
                                           nodes, containers)
                return

            elif utilization >= self.min_utilization:
                # Utilization is over what we've specified, lets recommend
                # this state.
                self._store_recommendation(source_id, container_nodes_group,
                                           nodes, containers)
                return

