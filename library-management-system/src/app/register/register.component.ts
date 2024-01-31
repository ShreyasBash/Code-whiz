import { Component } from '@angular/core';
import { Router } from '@angular/router'; // Correct import for Router
import { UserService } from '../user.service';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';


@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css']
})

export class RegisterComponent {
  registerForm: FormGroup;

  constructor(private userService: UserService, private router: Router, private fb: FormBuilder) {
    this.registerForm = this.fb.group({
      username: ['', [Validators.required,Validators.minLength(4)]],
      email: ['', [Validators.required, Validators.email]],
      password: ['', [Validators.required, Validators.minLength(6)]],
    });
  }

  accountAlreadyExists: string = '';
  
  registerUser() {

    this.accountAlreadyExists = '';

    this.userService.register(this.registerForm.value).subscribe(
      (response: any) => {
        console.log("Registered Successfully")
        // You can handle the response here, e.g., store user data in local storage and navigate to a different page
        localStorage.setItem('token', response.token);
        this.userService.login(this.registerForm.value).subscribe(
          (response: any) => {
            localStorage.setItem('token', response.token);
            console.log('Login successful', response);
            this.userService.navigation();
          } , (error) => {
            console.error("Login failed");
      }
        );
      },
      (error) => {
        // Handle registration error, e.g., display an error message to the user
        console.error('Registration failed:', error);
        if(error.status === 500) {
          this.accountAlreadyExists = "Account already exists";
        }
      }
    );
  }
}
