import { Prop, Schema, SchemaFactory } from "@nestjs/mongoose";
import { Document } from "mongoose";

@Schema({
    timestamps: true
})

export class Book extends Document{

    @Prop()
    image: string;

    @Prop({required: true, unique: true})
    title: string;

    @Prop({required: true})
    description: string;

    @Prop({required: true})
    author: string;

    @Prop({required: true})
    category: string;

    @Prop({enum: ['Available', 'Not available'], default: 'Available'})
    status: string;


}

export const BooksSchema = SchemaFactory.createForClass(Book)