import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { UserService } from '../user.service';
import { FormBuilder, FormGroup, FormControl, Validators } from '@angular/forms';



@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})


export class LoginComponent {

  loginForm: FormGroup;

  constructor(private userService: UserService, private router: Router, private fb: FormBuilder) {
  
  this.loginForm = this.fb.group({
    email: ['',[Validators.required, Validators.email]],
    password: ['',[Validators.required, Validators.minLength(6)]]
  });
}

incorrectPasswordError: string = '';
  

  loginUser() {
  
    this.incorrectPasswordError = '';
    this.userService.login(this.loginForm.value).subscribe(
      (response: any) => {
        // Successful login
      
        localStorage.setItem('token', response.token);
        console.log('Login successful', response);
        this.userService.navigation();
      },
      (error) => {
        // Handle login error, e.g., display an error message to the user
        
        console.error('Login failed:', error);
        if (error.status === 401) {
          this.incorrectPasswordError = 'Invalid email or password';
        }
      }
    );
  }
}
