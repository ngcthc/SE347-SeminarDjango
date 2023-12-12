from django.shortcuts import render
from django.http import HttpResponseRedirect
from .models import Book

# Create your views here.
def index(request):
    books = Book.objects.all()
    return render(request, 'book/index.html', {'books': books})

def create(request):
    if request.method == 'POST':
        title = request.POST['title']
        price = request.POST['price']
        # cach 1
        # book = Book.objects.create(title=title, price=price)
        # cach 2
        book = Book(title=title, price=price)
        book.save()
        return HttpResponseRedirect('/')
    else:
        return render(request, 'book/create.html')
