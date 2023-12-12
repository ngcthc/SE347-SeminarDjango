from django.db import models

# Create your models here.
class Book(models.Model):
    title = models.CharField(max_length=100)
    price = models.IntegerField(default=0)

    def __str__(self):
        return self.title