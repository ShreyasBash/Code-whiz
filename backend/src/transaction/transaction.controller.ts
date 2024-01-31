import { Controller, Post, Param, Body, Get, UseGuards } from '@nestjs/common';
import { TransactionService } from './transaction.service';
import { Transaction } from './schema/transaction.schema';
import { AuthGuard } from '@nestjs/passport';

@Controller('transaction')
export class TransactionController {
    constructor(private readonly transactionService: TransactionService) {}

    @Post('borrow/:userId/:bookId')
    async borrowBook(@Param('userId') userId: string, @Param('bookId') bookId: string): Promise<Transaction> {
      try {
        const transaction = this.transactionService.borrowBook(userId, bookId);
        console.log(transaction);
        return transaction;
      } catch(error) {
        console.log("Failed to borrow book",error);
        throw error;
      }
    }

    @Post('return/:id')
    async returnBook(@Param('id') id: string): Promise<Transaction> {
      try{
        const transaction = this.transactionService.returnBook(id);
        console.log(transaction);
        return transaction;
      } catch (error) {
        console.log("Error in returning the book", error);
        throw error;
      }
    }

    @Get('history/:userId')
  async viewTransactionHistory(@Param('userId') userId: string): Promise<Transaction[]> {
    try {
      const transaction = this.transactionService.transactionHistory(userId);
      return transaction;
    } catch (error) {
      console.log("Error in viewing history of transactions",error);
      throw error;
    }
  }
}
