import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ValidationPipe } from '@nestjs/common';
import * as multer from 'multer';
import { join } from 'path';
import * as express from 'express';


async function bootstrap() {
  const app = await NestFactory.create(AppModule, { cors: true });
  app.enableCors({
    origin: ['http://localhost:4200'],
    methods: ['GET','POST','PUT','DELETE'],
    credentials: true,
  });


  app.use(express.static(__dirname +'./images/'));

  app.use('/images', express.static('./images/'));

  app.useGlobalPipes(new ValidationPipe());
  await app.listen(3000);
}
bootstrap();
