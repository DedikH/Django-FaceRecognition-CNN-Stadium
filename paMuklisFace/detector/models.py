from django.db import models
from django import forms

class Pelaku(models.Model):
    id = models.AutoField(primary_key=True)
    nama = models.CharField(max_length=256)
    umur = models.CharField(max_length=256)
    kasus = models.CharField(max_length=256)
    tim_dukung = models.CharField(max_length=256)
    foto_pelaku = models.ImageField(upload_to='foto_pelaku/', null=True, blank=True)

    def __str__(self):
        return self.nama
