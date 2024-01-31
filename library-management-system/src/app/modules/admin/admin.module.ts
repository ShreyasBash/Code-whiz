import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { AdminRoutingModule } from './admin-routing.module';
import { AdminDashboardComponent } from './components/admin-dashboard/admin-dashboard.component';
import { HomeComponent } from './components/home/home.component';
import { BookComponent } from './components/book/book.component';
import { FormsModule } from '@angular/forms';
import { NgxPaginationModule } from 'ngx-pagination';
import { MatIconModule } from '@angular/material/icon';
import { ProfileComponent } from './components/profile/profile.component';
import { ReactiveFormsModule } from '@angular/forms';
import { AddComponent } from './components/add/add.component';
import { WebSocketService } from 'src/app/websocket.service';
import { ClientModule } from '../client/client.module';


@NgModule({
  declarations: [
    AdminDashboardComponent,
    HomeComponent,
    BookComponent,
    ProfileComponent,
    AddComponent,
  ],
  imports: [
    CommonModule,
    AdminRoutingModule,
    NgxPaginationModule,
    MatIconModule,
    FormsModule,
    ReactiveFormsModule,
  ],
  providers: [WebSocketService],
})
export class AdminModule { }
