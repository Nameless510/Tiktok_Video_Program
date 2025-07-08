#!/usr/bin/env python3
"""
Main test runner
Run all functional test files
"""

import os
import sys
import unittest
import time
from pathlib import Path

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("TikTok Video Feature Extraction System - Full Functional Test")
    print("=" * 60)
    
    # Get current directory
    current_dir = Path(__file__).parent
    test_files = [
        "test_video_processor.py",
        "test_audio_processor.py", 
        "test_frame_analyzer.py",
        "test_multimodal_extractor.py",
        "test_feature_extractor.py"
    ]
    
    # Statistics
    total_tests = 0
    total_failures = 0
    total_errors = 0
    test_results = []
    
    # Run each test file
    for test_file in test_files:
        test_path = current_dir / test_file
        
        if not test_path.exists():
            print(f"\n❌ Test file does not exist: {test_file}")
            continue
        
        print(f"\n{'='*20} Running {test_file} {'='*20}")
        
        # Import and run test
        try:
            # Dynamically import test module
            spec = unittest.util.spec_from_file_location(test_file[:-3], test_path)
            test_module = unittest.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)
            
            # Find test classes
            test_classes = []
            for name in dir(test_module):
                obj = getattr(test_module, name)
                if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj != unittest.TestCase:
                    test_classes.append(obj)
            
            if not test_classes:
                print(f"❌ No test class found in {test_file}")
                continue
            
            # Run tests
            for test_class in test_classes:
                print(f"\n--- Test class: {test_class.__name__} ---")
                
                # Create test suite
                test_suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                
                # Run tests
                runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
                result = runner.run(test_suite)
                
                # Record results
                total_tests += result.testsRun
                total_failures += len(result.failures)
                total_errors += len(result.errors)
                
                test_results.append({
                    'file': test_file,
                    'class': test_class.__name__,
                    'tests_run': result.testsRun,
                    'failures': len(result.failures),
                    'errors': len(result.errors),
                    'success': result.wasSuccessful()
                })
                
                # Show failure and error details
                if result.failures:
                    print(f"\n❌ Failed tests:")
                    for test, traceback in result.failures:
                        print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
                
                if result.errors:
                    print(f"\n❌ Error tests:")
                    for test, traceback in result.errors:
                        print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
        
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            total_errors += 1
    
    # Output summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Success: {total_tests - total_failures - total_errors}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%" if total_tests > 0 else "0%")
    
    # Detailed results
    print(f"\nDetailed results:")
    for result in test_results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['file']} - {result['class']}: "
              f"{result['tests_run']} tests, {result['failures']} failures, {result['errors']} errors")
    
    # Return overall success status
    overall_success = total_failures == 0 and total_errors == 0
    print(f"\nOverall result: {'✅ All tests passed' if overall_success else '❌ Some tests failed'}")
    
    return overall_success

def run_specific_test(test_name):
    """Run a specific test"""
    print(f"Running specific test: {test_name}")
    
    test_file = f"test_{test_name}.py"
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        print(f"❌ Test file does not exist: {test_file}")
        return False
    
    # Import and run specific test
    spec = unittest.util.spec_from_file_location(test_name, test_path)
    test_module = unittest.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # Find test classes
    test_classes = []
    for name in dir(test_module):
        obj = getattr(test_module, name)
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj != unittest.TestCase:
            test_classes.append(obj)
    
    if not test_classes:
        print(f"❌ No test class found in {test_file}")
        return False
    
    # Run tests
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class in test_classes:
        print(f"\n--- Test class: {test_class.__name__} ---")
        
        test_suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
    
    success = total_failures == 0 and total_errors == 0
    print(f"\nTest result: {'✅ Passed' if success else '❌ Failed'}")
    print(f"  Tests: {total_tests}, Failures: {total_failures}, Errors: {total_errors}")
    
    return success

def show_test_menu():
    """Show test menu"""
    print("\n" + "=" * 60)
    print("TikTok Video Feature Extraction System - Test Menu")
    print("=" * 60)
    print("1. Run all tests")
    print("2. Test Video Processor")
    print("3. Test Audio Processor")
    print("4. Test Frame Analyzer")
    print("5. Test Multimodal Extractor")
    print("6. Test Feature Extractor")
    print("0. Exit")
    print("=" * 60)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Command line argument mode
        if sys.argv[1] == "all":
            success = run_all_tests()
            sys.exit(0 if success else 1)
        else:
            success = run_specific_test(sys.argv[1])
            sys.exit(0 if success else 1)
    else:
        # Interactive mode
        while True:
            show_test_menu()
            choice = input("Please select test option (0-6): ").strip()
            
            if choice == "0":
                print("Exiting test")
                break
            elif choice == "1":
                run_all_tests()
            elif choice == "2":
                run_specific_test("video_processor")
            elif choice == "3":
                run_specific_test("audio_processor")
            elif choice == "4":
                run_specific_test("frame_analyzer")
            elif choice == "5":
                run_specific_test("multimodal_extractor")
            elif choice == "6":
                run_specific_test("feature_extractor")
            else:
                print("Invalid choice, please enter again")
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 