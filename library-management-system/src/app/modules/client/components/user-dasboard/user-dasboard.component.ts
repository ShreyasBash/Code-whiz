import { Component, OnInit } from '@angular/core';
import { WebSocketService } from 'src/app/websocket.service';

@Component({
  selector: 'app-user-dasboard',
  templateUrl: './user-dasboard.component.html',
  styleUrls: ['./user-dasboard.component.css'],
})
export class UserDasboardComponent implements OnInit {
  constructor(private webSocketService: WebSocketService) {}

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
    this.webSocketService.onBookBorrowed().subscribe((payload: any) => {
      this.addNotification(`${payload.bookname} is now unavailable`);
    });

    // Listen for book returned events
    this.webSocketService.onBookReturned().subscribe((payload: any) => {
      this.addNotification(`${payload.bookname} is now available`);
    });

    this.webSocketService.onBookCreated().subscribe((payload: any) => {
      this.addNotification(`New Book/n${payload.bookName} is now available`);
    });

    this.webSocketService.onBookDeleted().subscribe((payload: any) => {
      this.addNotification(`Book removed/n${payload.bookName} is removed from library`);
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
