# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2019-09-18 04:40
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('client', '0006_accuracymodel2'),
    ]

    operations = [
        migrations.CreateModel(
            name='AccuracyModel3',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('accuracy3', models.CharField(max_length=100)),
            ],
        ),
    ]
