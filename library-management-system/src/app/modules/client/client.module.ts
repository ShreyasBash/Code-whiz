import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { ClientRoutingModule } from './client-routing.module';
import { UserDasboardComponent } from './components/user-dasboard/user-dasboard.component';
import { MatIconModule } from '@angular/material/icon';
import { HomeComponent } from './components/home/home.component';
import { BorrowComponent } from './components/borrow/borrow.component';
import { HistoryComponent } from './components/history/history.component';
import { ProfileComponent } from './components/profile/profile.component';
import { CoursesComponent } from './components/courses/courses.component';
import { FormsModule } from '@angular/forms';
import { NgxPaginationModule } from 'ngx-pagination';
import { ReactiveFormsModule } from '@angular/forms';
import { WebSocketService } from 'src/app/websocket.service';

@NgModule({
  declarations: [
    UserDasboardComponent,
    HomeComponent,
    BorrowComponent,
    HistoryComponent,
    ProfileComponent,
    CoursesComponent
  ],
  imports: [
    CommonModule,
    ClientRoutingModule,
    MatIconModule,
    FormsModule,
    NgxPaginationModule,
    ReactiveFormsModule,
  ],
  providers: [WebSocketService],
})
export class ClientModule { }
