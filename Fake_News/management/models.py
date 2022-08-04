from django.db import models

# Create your models here.


class AdminModel(models.Model):
    newsid = models.IntegerField()
    title = models.CharField(max_length=20000000)
    text = models.CharField(max_length=100000000)
    label = models.CharField(max_length=100000000)