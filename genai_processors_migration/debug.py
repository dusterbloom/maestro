#!/usr/bin/env python3
"""
Simple debug script to check genai-processors API without async issues.
"""

import inspect

def check_processor_interface():
    """Check what methods the Processor class requires."""
    print("🔍 Checking Processor interface...")
    
    try:
        from genai_processors import processor
        print("✅ Imported genai_processors.processor")
        
        # Check abstract methods
        abstract_methods = getattr(processor.Processor, '__abstractmethods__', set())
        print(f"🔧 Abstract methods required: {abstract_methods}")
        
        # Check if __call__ exists
        if hasattr(processor.Processor, '__call__'):
            try:
                sig = inspect.signature(processor.Processor.__call__)
                print(f"📋 __call__ signature: {sig}")
            except:
                print("📋 __call__ exists but signature not accessible")
        
        # Check if call exists  
        if hasattr(processor.Processor, 'call'):
            try:
                sig = inspect.signature(processor.Processor.call)
                print(f"📋 call signature: {sig}")
            except:
                print("📋 call exists but signature not accessible")
        
        # List all methods
        methods = [name for name in dir(processor.Processor) if not name.startswith('_') or name in ['__call__', '__init__']]
        print(f"📝 Available methods: {methods}")
        
        return abstract_methods
        
    except ImportError as e:
        print(f"❌ Cannot import: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def check_content_api():
    """Check content API."""
    print(f"\n🔍 Checking content API...")
    
    try:
        from genai_processors import content_api
        print("✅ Imported genai_processors.content_api")
        
        # Check ProcessorPart
        if hasattr(content_api, 'ProcessorPart'):
            print("✅ ProcessorPart available")
            
            # Try to create one
            try:
                part = content_api.ProcessorPart(
                    data="test",
                    mimetype="text/plain"
                )
                print(f"✅ ProcessorPart created successfully")
                print(f"   Data: {part.data}")
                print(f"   Mimetype: {part.mimetype}")
                
                # Check metadata
                if hasattr(part, 'metadata'):
                    print(f"   Metadata: {part.metadata}")
                else:
                    print(f"   No metadata attribute")
                    
            except Exception as e:
                print(f"❌ ProcessorPart creation failed: {e}")
        
        # Check utility functions
        utils = []
        for name in ['is_audio', 'is_text', 'is_image']:
            if hasattr(content_api, name):
                utils.append(name)
        
        if utils:
            print(f"✅ Utility functions: {utils}")
        else:
            print(f"❌ No utility functions found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_streams():
    """Check streams API."""
    print(f"\n🔍 Checking streams API...")
    
    try:
        from genai_processors import streams
        print("✅ Imported genai_processors.streams")
        
        # Check stream_content function
        if hasattr(streams, 'stream_content'):
            print("✅ stream_content function available")
            
            try:
                # Test creating a stream (but don't iterate)
                stream = streams.stream_content(["hello"])
                print(f"✅ Stream created: {type(stream)}")
            except Exception as e:
                print(f"❌ Stream creation failed: {e}")
        else:
            print(f"❌ stream_content function not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_simple_processor():
    """Test creating a simple processor."""
    print(f"\n🧪 Testing simple processor creation...")
    
    try:
        from genai_processors import processor, content_api
        
        # Try with __call__ method
        class TestProcessor1(processor.Processor):
            async def __call__(self, content):
                async for part in content:
                    yield part
        
        try:
            proc1 = TestProcessor1()
            print("✅ Processor with __call__ method works")
        except Exception as e:
            print(f"❌ Processor with __call__ failed: {e}")
        
        # Try with call method
        class TestProcessor2(processor.Processor):
            async def call(self, content):
                async for part in content:
                    yield part
        
        try:
            proc2 = TestProcessor2()
            print("✅ Processor with call method works")
        except Exception as e:
            print(f"❌ Processor with call failed: {e}")
        
        # Try with both methods
        class TestProcessor3(processor.Processor):
            async def __call__(self, content):
                async for part in content:
                    yield part
            
            async def call(self, content):
                async for part in content:
                    yield part
        
        try:
            proc3 = TestProcessor3()
            print("✅ Processor with both methods works")
        except Exception as e:
            print(f"❌ Processor with both methods failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing processors: {e}")
        return False

def check_version():
    """Check version info."""
    print(f"🔍 Checking version...")
    
    try:
        import genai_processors
        
        # Try different version attributes
        for attr in ['__version__', 'version', 'VERSION']:
            if hasattr(genai_processors, attr):
                version = getattr(genai_processors, attr)
                print(f"✅ Version: {version}")
                return
        
        # Try package metadata
        try:
            import importlib.metadata
            version = importlib.metadata.version('genai-processors')
            print(f"✅ Version (pip): {version}")
        except:
            print(f"⚪ Version not determinable")
        
    except Exception as e:
        print(f"❌ Error checking version: {e}")

def main():
    """Run all checks."""
    print("🐛 Simple GenAI Processors Debug")
    print("=" * 40)
    
    check_version()
    abstract_methods = check_processor_interface()
    check_content_api()
    check_streams()
    test_simple_processor()
    
    print(f"\n🎯 Key Finding:")
    if abstract_methods:
        if 'call' in abstract_methods:
            print("   ✅ Use 'call' method in your processors")
        elif '__call__' in abstract_methods:
            print("   ✅ Use '__call__' method in your processors")
        else:
            print(f"   ⚠️  Unknown abstract methods: {abstract_methods}")
    else:
        print("   ❌ Could not determine required methods")
    
    print(f"\n💡 Quick Fix:")
    print("   Use the UniversalAudioProcessor which implements both methods")

if __name__ == "__main__":
    main()