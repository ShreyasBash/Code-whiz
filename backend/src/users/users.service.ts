import { Injectable, NotFoundException, UnauthorizedException } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { User } from './schema/users.schema';
import * as bcrypt from 'bcryptjs';
import { JwtService } from '@nestjs/jwt';
import { CreateUserDto } from './dto/createuser.dto';
import { LoginUserDto } from './dto/loginuser.dto';
@Injectable()
export class UsersService {
   constructor(@InjectModel(User.name) private readonly userModel: Model<User>, private jwtService: JwtService) {}
   
   async register(createUserDto: CreateUserDto): Promise<{ token: string }> {
      const { username, email, password, userType } = createUserDto
      const hashedPassword = await bcrypt.hash(password, 10)
      const user = await this.userModel.create({
         username,
         email,
         password: hashedPassword,
         userType,
      })

      const token = this.jwtService.sign({ id: user._id })

      return { token }
   }

   async loginuser(loginUserDto: LoginUserDto): Promise<{ token: string }>{
      const { email, password } = loginUserDto;
      const user = await this.userModel.findOne({ email })

      if(!user) {
         throw new UnauthorizedException('Invalid email or password!')
      }

      const isPassword = await bcrypt.compare(password, user.password)

      if(!isPassword) {
         throw new UnauthorizedException('Invalid email or password!')
      }

      const token = this.jwtService.sign({ id: user._id });
      return { token };
   }

   async getUserDetails(userId: string): Promise<User> {
      const user = await this.userModel.findById(userId);
      if (!user) {
         throw new NotFoundException('User not found');
      }
   
      user.password = undefined;
      return user;
   }

   async editUser(userId: string, updateUserDto: Partial<User>): Promise<User> {
      const updatedUser = await this.userModel.findByIdAndUpdate(userId, updateUserDto, {
         new: true,
      });
   
      if (!updatedUser) {
         throw new NotFoundException('User not found');
      }
   
      updatedUser.password = undefined;
   
      return updatedUser;
   }
   
   
}
