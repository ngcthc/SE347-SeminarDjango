from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
# Create your views here.
def members(request):
    template1 = loader.get_template('template1.html')
    return HttpResponse(template1.render())
    # return HttpResponse('Helu')