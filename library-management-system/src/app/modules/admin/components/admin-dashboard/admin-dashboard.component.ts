import { Component } from '@angular/core';
import { UserService } from 'src/app/user.service';
import { WebSocketService } from 'src/app/websocket.service';

@Component({
  selector: 'app-admin-dashboard',
  templateUrl: './admin-dashboard.component.html',
  styleUrls: ['./admin-dashboard.component.css']
})
export class AdminDashboardComponent {
 constructor(private userService: UserService, private webSocketService: WebSocketService) {}

 name = localStorage.getItem('name');
 public isNotificationVisible: boolean = true;
  public notifications: string[] = [];

 logout() {
  localStorage.removeItem('name');
  localStorage.removeItem('token');
  localStorage.removeItem('userId');
  this.webSocketService.disconnect();
}

ngOnInit() {
  this.webSocketService.connect();

  // Listen for book borrowed events
  this.webSocketService.onBookCreated().subscribe((payload: any) => {
    this.addNotification(`New Book: ${payload.bookName} is created`);
  });

  // Listen for book returned events
  this.webSocketService.onBookDeleted().subscribe((payload: any) => {
    this.addNotification(`${payload.bookName} is deleted`);
  });
}

closeNotification(index: number) {
  this.notifications.splice(index, 1);
}  

private addNotification(message: string) {
  this.notifications.push(message);

  // Automatically remove the notification after 10 seconds
  setTimeout(() => {
    const index = this.notifications.indexOf(message);
    if (index !== -1) {
      this.notifications.splice(index, 1);
    }
  }, 10000); // 10 seconds in milliseconds
}
}
