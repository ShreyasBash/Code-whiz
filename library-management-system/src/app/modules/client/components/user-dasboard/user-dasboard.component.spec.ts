import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UserDasboardComponent } from './user-dasboard.component';

describe('UserDasboardComponent', () => {
  let component: UserDasboardComponent;
  let fixture: ComponentFixture<UserDasboardComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [UserDasboardComponent]
    });
    fixture = TestBed.createComponent(UserDasboardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
