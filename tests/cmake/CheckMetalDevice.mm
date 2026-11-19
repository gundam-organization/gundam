#import <Metal/Metal.h>

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if( device == nil ){ return 1; }
    [device release];
    return 0;
  }
}
