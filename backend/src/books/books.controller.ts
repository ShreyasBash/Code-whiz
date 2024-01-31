import { Body, Controller, Delete, Get, NotFoundException, Param, Post, Put, Query, UploadedFile, UseGuards, UseInterceptors } from '@nestjs/common';
import { BooksService } from './books.service';
import { Book } from './schema/books.schema';
import { Query as ExpressQuery } from 'express-serve-static-core';
import { Express } from 'express'; 
import { CreateBookDto } from './dto/createbook.dto';
import { UpdateBookDto } from './dto/updatebook.dto';
import { AuthGuard } from '@nestjs/passport';
import { FileInterceptor } from '@nestjs/platform-express';
import * as path from 'path';
import { diskStorage } from 'multer';


@Controller('books')
export class BooksController {
    constructor(private booksService: BooksService) {}

    @Get()
    @UseGuards(AuthGuard())
    async getAllBooks(): Promise<Book[]> {
        try{
            const book = this.booksService.allBooks();
            return book
        } catch (error) {
            console.log("Failed to get all books",error);
            throw error;
        }
    }

    @Post('add')
    @UseInterceptors(FileInterceptor('image', {
        storage: diskStorage({
            destination: './images',
            filename: (req, file, callBack) => {
                const fileName = path.parse(file.originalname).name.replace(/\s/g, '') + Date.now();
                const extension = path.parse(file.originalname).ext;
                callBack(null, `${fileName}${extension}`);
            }
        })
    }))
    async addNewBook(@UploadedFile() image: Express.Multer.File , @Body() createBookDto: CreateBookDto): Promise<Book> {
        try {
            const book =  this.booksService.addBook(createBookDto, image);
            console.log(image);
            return book;

        } catch(error) {
            console.log("failed to add book", error);
            throw error;
        }

    }

    
    @Get('search')
    async searchBook(@Query() query: ExpressQuery): Promise<Book[]> {
        try{
            const book = await this.booksService.searchBook(query);
            console.log(book);
            return book;
        } catch(error) {
            console.log("Error finding book",error);
            throw error;
        }
    }

    @Get(':id')
    @UseGuards(AuthGuard())
    async getSelectedBook(@Param('id') id: string): Promise<Book> {
        try{
            const book = await this.booksService.selectBook(id);
            console.log(book);
            return book;
        } catch (error) {
            console.log("Failed to get Book by id", error);
            throw error;
        }
    } 

    @Put(':id')
    @UseGuards(AuthGuard())
    async updateBook(@Param('id') id: string, @Body() updateBookDto: UpdateBookDto): Promise<Book> {
        try{
            const book = await this.booksService.updateBook(id, updateBookDto);
            console.log(book);
            return book;
        } catch(error) {
            console.log("Failed to edit selected book",error);
            throw error;
        }
    }
    
    @Delete(':id')
    @UseGuards(AuthGuard())
    async deleteBook(@Param('id') id: string): Promise<Book> {
        try{ 
            const book = await this.booksService.deleteBook(id);
            console.log(book);
            return book;
        } catch (error) {
            console.log(error);
        }
    }

}
