// notifications.gateway.ts
import { WebSocketGateway, WebSocketServer, OnGatewayConnection, OnGatewayDisconnect, SubscribeMessage } from '@nestjs/websockets';
import { Server } from 'socket.io';

@WebSocketGateway({
  cors: {
    origin: 'http://localhost:4200',  
  }
})
export class WebsocketGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer() server: Server;

  handleConnection(client: any) {
    // Handle new WebSocket connection
    console.log(client.id);
    console.log("Connected");
  }

  handleDisconnect(client: any) {

    console.log(client.id);
    console.log("Disconnected");
  }

  @SubscribeMessage('borrowed')
  handleBorrowedEvent(client: any, payload: any) {
    // Handle book borrowed event
    this.server.emit('bookBorrowed', payload);
  }

  @SubscribeMessage('returned')
  handleReturnedEvent(client: any, payload: any) {
    // Handle book returned event
    this.server.emit('bookReturned', payload);
  }

  @SubscribeMessage('create')
  handleCreatedEvent(client: any, payload: any) {

    this.server.emit('bookCreated', payload);
  }

  @SubscribeMessage('deleted')
  handleDeletedEvent(client: any, payload: any) {

    this.server.emit('bookDeleted',payload );
  }
}
