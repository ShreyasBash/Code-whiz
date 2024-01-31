import { Prop, Schema, SchemaFactory } from "@nestjs/mongoose";
import * as mongoose from "mongoose";
import { User } from "src/users/schema/users.schema";
import { Book } from "src/books/schema/books.schema";

@Schema({
    timestamps: true
})
export class Transaction extends mongoose.Document {
    @Prop({ type: mongoose.Schema.Types.ObjectId, ref: 'User' })
    user: User;

    @Prop({ type: mongoose.Schema.Types.ObjectId, ref: 'Book' })
    book: Book;

    @Prop({ required: true, enum: [ 'Borrowed', 'Returned' ] })
    status: string;

    @Prop({ required: true})
    borrowedDate: string;

    @Prop()
    returnedDate: string; 
}

export const TransactionSchema = SchemaFactory.createForClass(Transaction);
