import { Component } from '@angular/core';
import { UserService } from 'src/app/user.service';
import { WebSocketService } from 'src/app/websocket.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {
  borrows: any[] = [];
  returns: any[] = [];
  pending: any[] = [];
  currentPage: number = 1;
  itemsPerPage: number = 5;

  constructor(private userService: UserService, private webSocketService: WebSocketService) {}

  ngOnInit() {
    this.getBorrowedBooks();
  }

  getBorrowedBooks() {
    this.borrows = [];
    this.returns = [];
    this.pending = [];
    this.userService.getBorrowedBooks().subscribe(
        (data: any) => {
          this.borrows = data;
          this.returns = data.filter((item: any) => item.status === "Returned");
          this.pending = data.filter((item: any) => item.status === 'Borrowed');

          console.log(data);
        },
        (error) => {
          console.error("Error fetching borrowed books", error);
        }
    );
  }

  returnBook(borrowId: string) {
    this.userService.returnBook(borrowId).subscribe(
      (data: any) => {
        console.log("Book returned Successfully", data);
        this.webSocketService.emitReturnedEvent({ bookname: data.book.title });
        window.location.reload();
      }, (error) => {
        console.error('Error returning book', error);
      }
    );
  }
}
