# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2019-09-18 03:56
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('client', '0003_fakerealmodel'),
    ]

    operations = [
        migrations.CreateModel(
            name='AccuracyModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('accuracy', models.CharField(max_length=100)),
            ],
        ),
    ]
