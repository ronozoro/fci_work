# -*- coding: utf-8 -*-
from odoo import models, fields


class AlgorithmImages(models.Model):
    _name = 'fci.algorithm.images'
    _rec_name = 'name'

    algorithm_id = fields.Many2one(comodel_name='fci.algorithm.design', string='Algorithm')
    name = fields.Char(string='Image Description', required=True)
    image = fields.Binary(string="Algorithm Image", required=True)


AlgorithmImages()
