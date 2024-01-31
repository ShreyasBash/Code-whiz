import { Component } from '@angular/core';
import { UserService } from 'src/app/user.service';
import { WebSocketService } from 'src/app/websocket.service';

@Component({
  selector: 'app-book',
  templateUrl: './book.component.html',
  styleUrls: ['./book.component.css']
})
export class BookComponent {
  searchKeyword: string = '';
  books: any[] = [];
  displayedBooks: any[] = [];
  itemsPerPage: number = 5; 
  currentPage: number = 1;
  totalPages: number = 1;
  constructor(private userService: UserService, private webSocketService: WebSocketService) {}

  ngOnInit() {

    const payload = localStorage.getItem('payload');
    if (payload) {
      this.webSocketService.emitDeletedEvent(JSON.parse(payload));

      localStorage.removeItem('payload');
    }
    
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

  deleteBook(bookId: string){
    this.userService.deleteBooks(bookId).subscribe(
      (data: any) => {
        console.log("Book deleted successfully", data);
        localStorage.setItem('payload', JSON.stringify({ bookName: data.title }));
        this.webSocketService.emitDeletedEvent({bookName: data.title});
        window.location.reload();
      }, (error) => {
        console.error("Failed to delete the book", error);
      }
    );
  }

}
