from django import forms
from login.models import Users

class LoginForm(forms.Form):
    username = forms.CharField(label='Tên người dùg', max_length=100)
    password = forms.CharField(label='Mật khẩu', widget=forms.PasswordInput)
    def clean_username(self):
        username = self.cleaned_data['username']
        if Users.objects.filter(username=username).exists():
            raise forms.ValidationError("Tên người dùng đã tồn tại")
        return username
    def clean_password(self):
        password = self.cleaned_data['password']
        if len(password) < 8:
            raise forms.ValidationError("Mật khẩu phải có ít nhất 8 ký tự")
        return password
    def save(self):
        Users.objects.create(username=self.cleaned_data['username'], password=self.cleaned_data['password'])