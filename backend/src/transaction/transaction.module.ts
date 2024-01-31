import { Module } from '@nestjs/common';
import { TransactionController } from './transaction.controller';
import { TransactionService } from './transaction.service';
import { MongooseModule } from '@nestjs/mongoose';
import { TransactionSchema } from './schema/transaction.schema';
import { UsersSchema } from 'src/users/schema/users.schema';
import { BooksSchema } from 'src/books/schema/books.schema';

@Module({
  imports: [MongooseModule.forFeature([
    { name: 'Transaction', schema: TransactionSchema },
    {name: 'User', schema: UsersSchema},
    { name: 'Book', schema: BooksSchema }])],
  controllers: [TransactionController],
  providers: [TransactionService]
})
export class TransactionModule {}
