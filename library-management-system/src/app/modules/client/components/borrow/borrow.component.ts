import { Component, OnInit } from '@angular/core';
import { UserService } from 'src/app/user.service';
import { WebSocketService } from 'src/app/websocket.service';
 // Import your user service

@Component({
  selector: 'app-borrow',
  templateUrl: './borrow.component.html',
  styleUrls: ['./borrow.component.css'],
})
export class BorrowComponent implements OnInit {
  searchKeyword: string = '';
  books: any[] = [];
  displayedBooks: any[] = [];
  itemsPerPage: number = 5; 
  currentPage: number = 1;
  totalPages: number = 1;
  constructor(private userService: UserService, private webSocketService: WebSocketService) {}

  ngOnInit() {
    // Fetch all books when the component initializes
    this.searchBooks();
  }

  fetchBooks() {
    this.books = [];
    this.userService.getBooks().subscribe(

      (data: any) => {
        this.books = data;
        this.totalPages = Math.ceil(this.books.length / this.itemsPerPage);
        this.updateDisplayedBooks();
      },
      (error) => {
        console.error("Error fetching books", error);
      }
    );
  }

  updateDisplayedBooks() {
    const startIndex = (this.currentPage - 1) * this.itemsPerPage;
    const endIndex = startIndex + this.itemsPerPage;
    this.displayedBooks = this.books.slice(startIndex, endIndex);
  }

  prevPage() {
    if (this.currentPage > 1) {
      this.currentPage--;
      this.updateDisplayedBooks();
    }
  }

  nextPage() {
    if (this.currentPage < this.totalPages) {
      this.currentPage++;
      this.updateDisplayedBooks();
    }
  }

  searchBooks() {
    if (this.searchKeyword !== '') {
      this.books = []; // Clear the current list of books

      this.userService.searchBook(this.searchKeyword).subscribe(
        (data: any) => {
          this.books = data;
          console.log(data);
        },
        (error) => {
          console.error('Failed to fetch the searched books', error);
        }
      );
    } else {
      // If the searchKeyword is empty, fetch all books
      this.fetchBooks();
    }
  }

  borrowBook(bookId: string) {
    // Call your backend API to borrow the book by its ID
    this.userService.borrowBook(bookId).subscribe(
      (response: any) => {
        // Handle the successful response from your backend if needed
        console.log('Book borrowed successfully', response);
        this.webSocketService.emitBorrowedEvent({ bookname: response.book.title });
        window.location.reload();
      },
      (error) => {
        console.error('Error borrowing book', error);
      }
    );
  }
}
