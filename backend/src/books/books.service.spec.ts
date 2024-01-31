import { Test, TestingModule } from "@nestjs/testing"
import { BooksService } from "./books.service"
import { getModelToken } from '@nestjs/mongoose'
import { Book } from "./schema/books.schema"
import { Model } from "mongoose";


describe('BooksService', () => {

    let booksService: BooksService;
    let model: Model<Book>;

    const mockBookService= {};

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [
                BooksService,
                {
                    provide: getModelToken(Book.name),
                    useValue: mockBookService
                },
            ],
        }).compile()

        booksService = module.get<BooksService>(BooksService);
        model = module.get<Model<Book>>(getModelToken(Book.name));

    });
});