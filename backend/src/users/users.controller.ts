import { Controller, Get, Post, Put, Body, UseGuards, Param } from "@nestjs/common";
import { UsersService } from "./users.service";
import { CreateUserDto } from "./dto/createuser.dto";
import { LoginUserDto } from "./dto/loginuser.dto";
import { AuthGuard } from "@nestjs/passport";
import { User } from "./schema/users.schema";
import { UpdateUserDto } from "./dto/updateuser.dto";

@Controller('users')
export class UsersController {
    constructor(private readonly usersService: UsersService) {}

    @Post('register')
    registerUser(@Body() createUserDto: CreateUserDto): Promise<{ token: string }> {
        try{
            const user = this.usersService.register(createUserDto);
            console.log(user);
            return user;
        } catch(error) {
            console.log("Failed to register user",error);
            throw error;
        }
    }

    @Post('login')
    loginUser(@Body() loginUserDto: LoginUserDto): Promise<{ token: string }> {
        try{
            const user = this.usersService.loginuser(loginUserDto);
            console.log(user);
            return user;
        } catch(error) {
            console.log("Failed to login",error);
            throw error;
        }
    }

    @Get(':id')
    @UseGuards(AuthGuard())
    getUser(@Param('id') id: string): Promise<User> {
        try {
            const user = this.usersService.getUserDetails(id);
            console.log(user);
            return user;
        } catch (error) {
            console.log("Failed to fetch user details", error);
            throw error;
        }
    }

    @Put(':id')
    updateUser(@Param('id') id: string, @Body() updateUserDto: UpdateUserDto): Promise<User> {
        try{
            const user = this.usersService.editUser(id, updateUserDto);
            console.log(user);
            return user;
        } catch(error) {
            console.log("Failed to edit user details",error);
            throw error;
        }
    }

}
