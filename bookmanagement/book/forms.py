from django import forms
from .models import Book

class BookForm(forms.Form):
    title = forms.CharField(label='Ten sach', max_length=100)
    price = forms.IntegerField(label='Gia sach')

    def clean_title(self):
        title = self.cleaned_data['title']
        if len(title) < 5:
            raise forms.ValidationError('Title must be more than 5 characters')
        return title

    def clean_price(self):
        price = self.cleaned_data['price']
        if price < 1000:
            raise forms.ValidationError('Price must be more than 1000')
        return price
    
    def save(self):
        title = self.cleaned_data['title']
        price = self.cleaned_data['price']
        book = Book.objects.create(title=title, price=price)
        book.save()
        return book