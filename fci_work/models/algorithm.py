# -*- coding: utf-8 -*-
import base64
import random
from collections import defaultdict,namedtuple
from heapq import *

import io
import itertools
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from odoo import models, fields, api
from odoo.exceptions import Warning
from odoo.tools.translate import _

parent = dict()
rank = dict()
inf = float('inf')
Edge = namedtuple('Edge', 'start, end, cost')


def make_edge(start, end, cost=1):
    return Edge(start, end, cost)


class MaxFlow():
    @staticmethod
    def depth_first_search(graph, source, sink):
        undirected = graph.to_undirected()
        explored = {source}
        stack = [(source, 0, dict(undirected[source]))]

        while stack:
            v, _, neighbours = stack[-1]
            if v == sink:
                break

            # search the next neighbour
            while neighbours:
                u, e = neighbours.popitem()
                if u not in explored:
                    break
            else:
                stack.pop()
                continue

            # current flow and capacity
            in_direction = graph.has_edge(v, u)
            capacity = e['capacity']
            flow = e['flow']
            neighbours = dict(undirected[u])

            # increase or redirect flow at the edge
            if in_direction and flow < capacity:
                stack.append((u, capacity - flow, neighbours))
                explored.add(u)
            elif not in_direction and flow:
                stack.append((u, flow, neighbours))
                explored.add(u)

        # (source, sink) path and its flow reserve
        reserve = min((f for _, f, _ in stack[1:]), default=0)
        path = [v for v, _, _ in stack]

        return path, reserve

    def ford_fulkerson(self, graph, source, sink, flow_to_database=None, record=None):
        flow, path = 0, True

        while path:
            # search for path with flow reserve
            path, reserve = self.depth_first_search(graph, source, sink)
            flow += reserve

            # increase flow along the path
            for v, u in zip(path, path[1:]):
                if graph.has_edge(v, u):
                    graph[v][u]['flow'] += reserve
                else:
                    graph[u][v]['flow'] -= reserve

            # show current results
            if callable(flow_to_database):
                flow_to_database(graph, path, reserve, flow, record)

    @staticmethod
    def draw_graph(graph):
        plt.figure(figsize=(20, 8))
        plt.axis('off')
        layout = nx.spring_layout(graph, k=1, iterations=20)
        nx.draw_networkx_nodes(graph, layout, node_color='gray', node_size=600)
        nx.draw_networkx_edges(graph, layout, edge_color='black')
        nx.draw_networkx_labels(graph, layout, font_color='white')

        for u, v, e in graph.edges(data=True):
            label = '{}/{}'.format(e['flow'], e['capacity'])
            color = 'green' if e['flow'] < e['capacity'] else 'red'
            x = layout[u][0] * .6 + layout[v][0] * .4
            y = layout[u][1] * .6 + layout[v][1] * .4
            plt.text(x, y, label, size=16, color=color, horizontalalignment='center', verticalalignment='center')
        buffered = io.BytesIO()
        plt.savefig(buffered, format="PNG")
        image_value = base64.b64encode(buffered.getvalue()).decode('utf8')
        return image_value

    def flow_to_database(self, graph, path, reserve, flow, record):
        text = 'flow increased by %s at path %s and current flow is %s' % (reserve, path, flow)
        image_src = self.draw_graph(graph)
        record.env['fci.algorithm.images'].sudo().create({
            'name': text,
            'algorithm_id': record.id,
            'image': image_src
        })


class Graph:
    def __init__(self, edges):
        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [2, 3]]
        if wrong_edges:
            raise ValueError('Wrong edges data: {}'.format(wrong_edges))

        self.edges = [make_edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set(
            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def get_node_pairs(self, n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs

    def remove_edge(self, n1, n2, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                self.edges.remove(edge)

    def add_edge(self, n1, n2, cost=1, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        for edge in self.edges:
            if [edge.start, edge.end] in node_pairs:
                return ValueError('Edge {} {} already exists'.format(n1, n2))

        self.edges.append(Edge(start=n1, end=n2, cost=cost))
        if both_ends:
            self.edges.append(Edge(start=n2, end=n1, cost=cost))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))
            neighbours[edge.end].add((edge.start, edge.cost))
        return neighbours

    def dijkstra(self, source, dest, record):
        assert source in self.vertices, 'Such source node doesn\'t exist'
        distances = {vertex: inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()
        images = []
        while vertices:

            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])
            vertices.remove(current_vertex)
            if distances[current_vertex] == inf:
                break

            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    last_cost = alternative_route
                    previous_vertices[neighbour] = {'item': current_vertex, 'cost': alternative_route,
                                                    'current_root_cost': cost}
                    G = nx.Graph()
                    for key, value in previous_vertices.items():
                        if value is not None:
                            G.add_edge(key, value.get('item'), weight=value.get('cost'))
                    pos = nx.spring_layout(G)
                    nx.draw_networkx_nodes(G, pos, node_color='gray', node_size=500, alpha=0.8)
                    nx.draw_networkx_edges(G, pos, width=3.0, alpha=0.5, edge_color='blue')
                    nx.draw_networkx_labels(G, pos, font_size=16)
                    edge_labels = nx.get_edge_attributes(G, 'weight')
                    nx.draw_networkx_edge_labels(G, pos, edge_labels)
                    buffered = io.BytesIO()
                    plt.savefig(buffered, format="PNG")
                    image_value = base64.b64encode(buffered.getvalue()).decode('utf8')
                    images.append({
                        'name': 'Building Dijstra',
                        'algorithm_id': record.id,
                        'image': image_value
                    })
                    plt.figure(figsize=(20, 8))
        return images


class Algorithm(models.Model):
    _name = 'fci.algorithm.design'
    _rec_name = 'algorithm'
    algorithm = fields.Selection(selection=[('kruskal', 'Kruskal'),
                                            ('prim', 'Prim'),
                                            ('dijkstra', 'Dijkstra'),
                                            ('max_flow', 'Max Flow')], string='Algorithm', default='kruskal',
                                 required=True)
    start_node = fields.Integer(string='Start Node', default=0)
    number_of_nodes = fields.Integer(string='No. of Nodes', default=3, required=True)
    edge_ids = fields.One2many(comodel_name='algorithm.design.edges', inverse_name='algorithm_id', string='Edges')
    image_ids = fields.One2many(comodel_name='fci.algorithm.images', inverse_name='algorithm_id', string='Images')
    total_cost = fields.Float()
    start_item = fields.Integer(string='Start From')
    end_item = fields.Integer(string='To')

    @api.constrains('number_of_nodes')
    def _constraint_nodes(self):
        for rec in self:
            if rec.number_of_nodes < 3:
                raise Warning(_("Number of nodes should be more than 2"))

    @api.constrains('start_node', 'number_of_nodes')
    def _constraint_start_node(self):
        for rec in self:
            if rec.start_node < 0 or rec.start_node not in list(range(0, rec.number_of_nodes)):
                raise Warning(_("Start Node Is not valid form Prim Algorithm"))

    @api.constrains('edge_ids', 'number_of_nodes')
    def _constraint_edges(self):
        for rec in self:
            for item in rec.edge_ids:
                if item.from_node not in range(0, rec.number_of_nodes):
                    raise Warning(_("Node %s is not in the Nodes range %s") % (
                        item.from_node, list(range(0, rec.number_of_nodes))))
                if item.to_node not in range(0, rec.number_of_nodes):
                    raise Warning(
                        _("Node %s is not in the Nodes range %s") % (item.to_node, list(range(0, rec.number_of_nodes))))

    def clear_data(self):
        self.env.cr.execute('delete from algorithm_design_edges')
        self.env['algorithm.design.edges'].invalidate_cache()
        self.env.cr.execute('delete from fci_algorithm_images')
        self.env['fci.algorithm.images'].invalidate_cache()

    def create_random_edges(self):
        for record in self:
            record.clear_data()
            dynamic_edges = list(itertools.combinations(range(record.number_of_nodes), 2))
            range_weights = range(1, len(dynamic_edges))
            my_edge_list = [(random.choice(range_weights), item[0], item[1])
                            for item in dynamic_edges]
            random.shuffle(my_edge_list)
            create_list = [{
                'from_node': item[1],
                'to_node': item[2],
                'weight': item[0],
                'algorithm_id': record.id
            } for item in my_edge_list]
            for item in create_list:
                self.env['algorithm.design.edges'].create(item)
        return True

    def make_set(self, vertice):
        parent[vertice] = vertice
        rank[vertice] = 0

    def find(self, vertice):
        if parent[vertice] != vertice:
            parent[vertice] = self.find(parent[vertice])
        return parent[vertice]

    def union(self, vertice1, vertice2):
        root1 = self.find(vertice1)
        root2 = self.find(vertice2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
            if rank[root1] == rank[root2]: rank[root2] += 1

    def get_final_dijkstra(self, g):
        G = nx.Graph()
        for i in g:
            G.add_edge(i[0], i[1], weight=i[2])

        pos = nx.spring_layout(G)
        plt.xticks([])
        plt.yticks([])
        nx.draw_networkx_nodes(G, pos, node_color='gray', node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=3.0, alpha=0.5, edge_color='blue')
        nx.draw_networkx_labels(G, pos, font_size=16)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        path1 = nx.dijkstra_path(G, 1, 5)
        length1 = nx.dijkstra_path_length(G, 1, 5)
        answer = []
        for i in range(0, len(path1) - 1):
            answer.append((path1[i], path1[i + 1]))
        nx.draw_networkx_edges(G, pos, edgelist=answer, width=3.0, alpha=0.5, edge_color='black')
        buffered = io.BytesIO()
        plt.savefig(buffered, format="PNG")
        image_value = base64.b64encode(buffered.getvalue()).decode('utf8')
        plt.figure(figsize=(20, 8))
        img_dict={
            'name': 'Final Graph of Dijstra',
            'algorithm_id': self.id,
            'image': image_value
        }
        return [img_dict], length1

    def kruskal(self, nodes, edges):
        minimum_spanning_tree = set()
        edges.sort()
        images = []
        G = nx.Graph()
        for record in edges:
            G.add_edge(record[1], record[2], weight=record[0])
        pos = nx.spring_layout(G, k=20)
        nx.draw_networkx_nodes(G, pos, node_size=600)
        nx.draw_networkx_edges(G, pos, width=2)
        nx.draw_networkx_labels(G, pos)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
        buffered = io.BytesIO()
        plt.savefig(buffered, format="PNG")
        image_value = base64.b64encode(buffered.getvalue()).decode('utf8')
        images.append({
            'name': 'The Graph before applying %s Algorithm' % ('Kurskal' if self.algorithm == 'kruskal' else 'Prim'),
            'algorithm_id': self.id,
            'image': image_value
        })
        plt.figure(figsize=(20, 8))
        for vertice in nodes:
            self.make_set(vertice)
        for edge in edges:
            weight, vertice1, vertice2 = edge
            if self.find(vertice1) != self.find(vertice2):
                G = nx.Graph()
                self.union(vertice1, vertice2)
                minimum_spanning_tree.add(edge)
                for item in minimum_spanning_tree:
                    G.add_edge(item[1], item[2], weight=item[0])
                pos = nx.spring_layout(G, k=7)
                nx.draw_networkx_nodes(G, pos, node_size=900)
                nx.draw_networkx_edges(G, pos, width=6)
                nx.draw_networkx_labels(G, pos)
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
                buffered = io.BytesIO()
                plt.savefig(buffered, format="PNG")
                image_value = base64.b64encode(buffered.getvalue()).decode('utf8')
                images.append({
                    'name': 'Adding Edge From %s to %s with weight %s' % (edge[1], edge[2], edge[0]),
                    'algorithm_id': self.id,
                    'image': image_value
                })
                plt.figure(figsize=(20, 8))
        return sorted(minimum_spanning_tree), images

    def prim(self, nodes, edges, start):
        images = []
        G = nx.Graph()
        for record in edges:
            G.add_edge(record[1], record[2], weight=record[0])
        pos = nx.spring_layout(G, k=7)
        nx.draw_networkx_nodes(G, pos, node_size=900)
        nx.draw_networkx_edges(G, pos, width=6)
        nx.draw_networkx_labels(G, pos)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
        buffered = io.BytesIO()
        plt.savefig(buffered, format="PNG")
        image_value = base64.b64encode(buffered.getvalue()).decode('utf8')
        images.append({
            'name': 'The Graph before applying %s Algorithm' % ('Kurskal' if self.algorithm == 'kruskal' else 'Prim'),
            'algorithm_id': self.id,
            'image': image_value
        })
        plt.figure(figsize=(20, 8))
        conn = defaultdict(list)
        minimum_spanning_tree = []
        for c, n1, n2 in edges:
            conn[n1].append((c, n1, n2))
            conn[n2].append((c, n2, n1))
        used = {start}
        usable_edges = conn[nodes[start]][:]
        heapify(usable_edges)
        while usable_edges:
            G = nx.Graph()
            edge_list_1 = []
            edge_list_2 = []
            for record in usable_edges:
                G.add_edge(record[1], record[2], weight=record[0], color='r')
                edge_list_1.append([record[1], record[2]])
            for item in minimum_spanning_tree:
                G.add_edge(item[1], item[2], weight=item[0], color='b')
                edge_list_2.append([item[1], item[2]])
            pos = nx.spring_layout(G, k=7)
            nx.draw_networkx_nodes(G, pos, node_size=900)
            nx.draw_networkx_edges(G, pos, width=6, edgelist=edge_list_1, edge_color='red')
            nx.draw_networkx_edges(G, pos, width=6, edgelist=edge_list_2, edge_color='black')
            nx.draw_networkx_labels(G, pos)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
            buffered = io.BytesIO()
            plt.savefig(buffered, format="PNG")
            image_value = base64.b64encode(buffered.getvalue()).decode('utf8')
            images.append({
                'name': 'Step in build algorithm',
                'algorithm_id': self.id,
                'image': image_value
            })
            plt.figure(figsize=(20, 8))
            cost, n1, n2 = heappop(usable_edges)
            if n2 not in used:
                G = nx.Graph()
                used.add(n2)
                minimum_spanning_tree.append((cost, n1, n2))
                for item in minimum_spanning_tree:
                    G.add_edge(item[1], item[2], weight=item[0])
                pos = nx.spring_layout(G, k=7)
                nx.draw_networkx_nodes(G, pos, node_size=900)
                nx.draw_networkx_edges(G, pos, width=6)
                nx.draw_networkx_labels(G, pos)
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
                buffered = io.BytesIO()
                plt.savefig(buffered, format="PNG")
                image_value = base64.b64encode(buffered.getvalue()).decode('utf8')
                images.append({
                    'name': 'Adding Edge From %s to %s with weight %s' % (n1, n2, cost),
                    'algorithm_id': self.id,
                    'image': image_value
                })
                plt.figure(figsize=(20, 8))
                for e in conn[n2]:
                    if e[2] not in used:
                        heappush(usable_edges, e)
        return minimum_spanning_tree, images

    def call_algorithm(self):
        self.env.cr.execute('delete from fci_algorithm_images')
        self.env['fci.algorithm.images'].invalidate_cache()
        for record in self:
            edges = []
            for item in record.edge_ids:
                edges.append((item.weight, item.from_node, item.to_node))
            if record.algorithm == 'kruskal':
                nodes = list(range(0, record.number_of_nodes))
                edges = list(set(edges))
                mst, images = self.kruskal(nodes, edges)
                for image in images:
                    self.env['fci.algorithm.images'].create(image)
                self.total_cost = sum(tree[0] for tree in mst)
            elif record.algorithm == 'prim':
                nodes = list(range(0, record.number_of_nodes))
                edges = set(edges)
                mst, images = self.prim(nodes, edges, record.start_node)
                for image in images:
                    self.env['fci.algorithm.images'].create(image)
                self.total_cost = sum(tree[0] for tree in mst)
            elif record.algorithm == 'dijkstra':
                g = []
                for edge_item in record.edge_ids:
                    g.append((edge_item.from_node, edge_item.to_node, edge_item.weight))
                graph = Graph(g)
                graph_images = graph.dijkstra(record.start_item, record.end_item, record)
                final_image, cost = self.get_final_dijkstra(g)
                all_images = graph_images + final_image
                for image in all_images:
                    self.env['fci.algorithm.images'].create(image)
                self.total_cost = cost
            else:
                graph = nx.DiGraph()
                graph.add_nodes_from(list(range(0, record.number_of_nodes)))
                list_max_flow = []
                for edge_item in record.edge_ids:
                    list_max_flow.append(
                        (edge_item.from_node, edge_item.to_node, {'capacity': edge_item.weight, 'flow': 0}))

                graph.add_edges_from(list_max_flow)
                max_flow_obj = MaxFlow()
                max_flow_obj.ford_fulkerson(graph, record.start_item, record.end_item, max_flow_obj.flow_to_database,
                                            self)
        return True


Algorithm()


class AlgorithmEdges(models.Model):
    _name = 'algorithm.design.edges'
    algorithm_id = fields.Many2one(comodel_name='fci.algorithm.design', string='Algorithm')
    from_node = fields.Integer(string='From Node', required=True)
    to_node = fields.Integer(string='To Node', required=True)
    weight = fields.Float(string='Edge Weight', required=True, default=1.0)

    @api.constrains('weight')
    def _constraint_weight(self):
        for rec in self:
            if rec.weight <= 0:
                raise Warning(
                    _("Weight Can't be less than 0 for the edge from %s to %s") % (rec.from_node, rec.to_node))


AlgorithmEdges()
