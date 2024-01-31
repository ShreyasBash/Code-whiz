import { IsEnum, IsNotEmpty, IsOptional, IsString } from 'class-validator';



export class CreateBookDto {

    @IsOptional()
    readonly image: string;
    
    @IsNotEmpty()
    @IsString()
    readonly title: string;

    @IsNotEmpty()
    @IsString()
    readonly description: string;

    @IsNotEmpty()
    @IsString()
    readonly author: string;

    @IsNotEmpty()
    @IsString()
    readonly category: string;
}