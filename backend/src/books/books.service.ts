import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Book } from './schema/books.schema';
import * as mongoose from 'mongoose';
import { Query } from 'express-serve-static-core';
import { CreateBookDto } from './dto/createbook.dto';
import { UpdateBookDto } from './dto/updatebook.dto';

@Injectable()
export class BooksService {
    constructor(@InjectModel(Book.name) private readonly bookModel: mongoose.Model<Book>) {}
    
    async allBooks(): Promise<Book[]> {
        const allBooks = await this.bookModel.find().sort({createdAt: -1});
        return allBooks;
    }

    async addBook(createBookDto: CreateBookDto, image: Express.Multer.File): Promise<Book> {
        try {
          const { title, description, author, category } = createBookDto;
          let newBook: Book;
      
          if (image) {
            const imageFileName = image.filename; 
            newBook = await this.bookModel.create({
              title,
              description,
              author,
              category,
              image: imageFileName,
            });
          } else {
            newBook = await this.bookModel.create(createBookDto);
          }
      
          return newBook;
        } catch (error) {
          console.error('Error during image upload:', error);
          throw new BadRequestException('Failed to upload the image.');
        }
      }
      

    async selectBook(id: string): Promise<Book> {
        const isValidId = mongoose.isValidObjectId(id)

        if (!isValidId) {
            throw new BadRequestException('Please enter correct Id!');
        }
        
        const selectedBook = await this.bookModel.findById(id);

        if (!selectedBook) {
            throw new NotFoundException('Book not found');
        }

        return  selectedBook ;
    }

    async updateBook(id: string, updateBookDto: UpdateBookDto): Promise<Book> {
        const updatedBook = await this.bookModel.findByIdAndUpdate(id, updateBookDto, {
            new: true,
            runValidators: true,
        });
        return updatedBook;
    }

    async deleteBook(id: string): Promise<Book> {
        const deletedBook = await this.bookModel.findByIdAndDelete(id);
        return deletedBook;
    }

    async searchBook(query: Query): Promise<Book[]> {

        const keyword = query.keyword ? {
            title: {
                $regex: query.keyword,
                $options: 'i'
            }
        } : {}
        const books = await this.bookModel.find({ ...keyword }).sort({createdAt: -1}); 
        return books;
    }


}
