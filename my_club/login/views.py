from django.http import HttpResponseRedirect
from django.shortcuts import render
from .form import LoginForm
from .models import Users

def login_view(request):
    form = LoginForm()
    if request.method == 'POST':
        form = LoginForm(request.POST) # tạo một thể hiện biểu mẫu và điền nó với dữ liệu từ yêu cầu:
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/display/') 
    return render(request, 'login.html', {'form': form})

def display(request):
    return render(request, 'display.html', {'Users': Users.objects.all()})