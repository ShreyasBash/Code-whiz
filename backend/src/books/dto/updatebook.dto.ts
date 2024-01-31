import { IsEnum, IsOptional, IsString } from "class-validator";

export class UpdateBookDto {

    @IsOptional()
    readonly image: Express.Multer.File;

    @IsOptional()
    @IsString()
    readonly title: string;

    @IsOptional()
    @IsString()
    readonly description: string;

    @IsOptional()
    @IsString()
    readonly author: string;

    @IsOptional()
    @IsString()
    readonly category: string;
}