import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { UserDasboardComponent } from './components/user-dasboard/user-dasboard.component';
import { HomeComponent } from './components/home/home.component';
import { BorrowComponent } from './components/borrow/borrow.component';
import { HistoryComponent } from './components/history/history.component';
import { ProfileComponent } from './components/profile/profile.component';
import { AuthenticationGuard } from 'src/app/guards/authentication.guard';
import { CoursesComponent } from './components/courses/courses.component';

const routes: Routes = [
  {path: '', redirectTo:'home', pathMatch: 'full',},
  {path: 'home', component: HomeComponent},
  {path:'book', component: BorrowComponent},
  {path:'history', component:HistoryComponent},
  {path: 'profile', component:ProfileComponent},
  {path: 'courses', component:CoursesComponent}
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class ClientRoutingModule { }
