# -*- coding: utf-8 -*-
import base64
import io
import itertools
import random
from collections import defaultdict
from heapq import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from odoo import models, fields, api
from odoo.exceptions import Warning
from odoo.tools.translate import _

parent = dict()
rank = dict()


class Algorithm(models.Model):
    _name = 'fci.algorithm.design'
    _rec_name = 'algorithm'
    algorithm = fields.Selection(selection=[('kruskal', 'Kruskal'),
                                            ('prim', 'Prim')], string='Algorithm', default='kruskal', required=True)
    start_node = fields.Integer(string='Start Node', default=0)
    number_of_nodes = fields.Integer(string='No. of Nodes', default=3, required=True)
    edge_ids = fields.One2many(comodel_name='algorithm.design.edges', inverse_name='algorithm_id', string='Edges')
    image_ids = fields.One2many(comodel_name='fci.algorithm.images', inverse_name='algorithm_id', string='Images')
    total_cost = fields.Float()

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
        nx.draw_networkx_edge_labels(G, pos=pos,edge_labels=edge_labels)
        buffered = io.BytesIO()
        plt.savefig(buffered, format="PNG")
        image_value = base64.b64encode(buffered.getvalue()).decode('utf8')
        images.append({
            'name': 'The Graph before applying %s Algorithm' % ('Kurskal' if self.algorithm == 'kruskal' else 'Prim'),
            'algorithm_id': self.id,
            'image': image_value
        })
        plt.figure(figsize=(15,15))
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
                plt.figure(figsize=(15,15))
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
        plt.figure(figsize=(15,15))
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
            edge_list_1=[]
            edge_list_2=[]
            for record in usable_edges:
                G.add_edge(record[1], record[2], weight=record[0],color='r')
                edge_list_1.append([record[1],record[2]])
            for item in minimum_spanning_tree:
                G.add_edge(item[1], item[2], weight=item[0],color='b')
                edge_list_2.append([item[1], item[2]])
            pos = nx.spring_layout(G, k=7)
            nx.draw_networkx_nodes(G, pos, node_size=900)
            nx.draw_networkx_edges(G, pos, width=6,edgelist=edge_list_1,edge_color='red')
            nx.draw_networkx_edges(G, pos, width=6,edgelist=edge_list_2,edge_color='black')
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
            plt.figure(figsize=(15,15))
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
                plt.figure(figsize=(15,15))
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
            else:
                nodes = list(range(0, record.number_of_nodes))
                edges = set(edges)
                mst, images = self.prim(nodes, edges, record.start_node)
                for image in images:
                    self.env['fci.algorithm.images'].create(image)
                self.total_cost = sum(tree[0] for tree in mst)
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
