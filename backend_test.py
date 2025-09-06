import requests
import sys
import json
import io
import pandas as pd
from datetime import datetime

class TallyProcessorAPITester:
    def __init__(self, base_url="https://tally-processor.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.client_id = None
        self.file_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'} if not files else {}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, timeout=30)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=30)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test API health check"""
        success, response = self.run_test(
            "API Health Check",
            "GET",
            "",
            200
        )
        return success

    def test_create_client(self):
        """Test client creation"""
        client_data = {
            "name": "Test Company",
            "email": "test@example.com"
        }
        
        success, response = self.run_test(
            "Create Client",
            "POST",
            "clients",
            200,
            data=client_data
        )
        
        if success and 'id' in response:
            self.client_id = response['id']
            print(f"   Created client with ID: {self.client_id}")
            return True
        return False

    def test_get_clients(self):
        """Test getting all clients"""
        success, response = self.run_test(
            "Get All Clients",
            "GET",
            "clients",
            200
        )
        
        if success:
            print(f"   Found {len(response)} clients")
            return True
        return False

    def test_get_client_by_id(self):
        """Test getting specific client"""
        if not self.client_id:
            print("❌ Skipping - No client ID available")
            return False
            
        success, response = self.run_test(
            "Get Client by ID",
            "GET",
            f"clients/{self.client_id}",
            200
        )
        return success

    def create_test_csv(self):
        """Create a test CSV file for upload"""
        data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Description': ['SALARY CREDIT', 'ATM WITHDRAWAL', 'ONLINE TRANSFER'],
            'Amount': [50000, -2000, -5000],
            'Balance': [50000, 48000, 43000]
        }
        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        return csv_content.encode('utf-8')

    def test_file_upload(self):
        """Test file upload functionality"""
        try:
            # Create test CSV content
            csv_content = self.create_test_csv()
            
            files = {
                'file': ('test_statement.csv', csv_content, 'text/csv')
            }
            
            success, response = self.run_test(
                "File Upload",
                "POST",
                "upload-statement",
                200,
                files=files
            )
            
            if success and 'file_id' in response:
                self.file_id = response['file_id']
                print(f"   Uploaded file with ID: {self.file_id}")
                print(f"   Headers detected: {response.get('headers', [])}")
                print(f"   Suggested mapping: {response.get('suggested_mapping', {})}")
                return True
            return False
            
        except Exception as e:
            print(f"❌ File upload failed: {str(e)}")
            return False

    def test_confirm_mapping(self):
        """Test column mapping confirmation"""
        if not self.file_id or not self.client_id:
            print("❌ Skipping - No file ID or client ID available")
            return False
            
        mapping_data = {
            "date_column": "Date",
            "narration_column": "Description", 
            "amount_column": "Amount",
            "balance_column": "Balance",
            "statement_format": "single_amount_crdr"
        }
        
        success, response = self.run_test(
            "Confirm Column Mapping",
            "POST",
            f"confirm-mapping/{self.file_id}?client_id={self.client_id}",
            200,
            data=mapping_data
        )
        return success

    def test_regex_patterns(self):
        """Test regex pattern creation"""
        if not self.client_id:
            print("❌ Skipping - No client ID available")
            return False
            
        pattern_data = {
            "client_id": self.client_id,
            "pattern": ".*SALARY.*",
            "ledger_name": "Salary Income",
            "sample_narrations": ["SALARY CREDIT", "MONTHLY SALARY"]
        }
        
        success, response = self.run_test(
            "Create Regex Pattern",
            "POST",
            "regex-patterns",
            200,
            data=pattern_data
        )
        return success

    def test_get_client_patterns(self):
        """Test getting client regex patterns"""
        if not self.client_id:
            print("❌ Skipping - No client ID available")
            return False
            
        success, response = self.run_test(
            "Get Client Regex Patterns",
            "GET",
            f"regex-patterns/{self.client_id}",
            200
        )
        return success

    def test_ai_improve_regex(self):
        """Test AI regex improvement"""
        ai_request = {
            "narrations": ["SALARY CREDIT", "MONTHLY SALARY TRANSFER"],
            "existing_regex": ".*SALARY.*",
            "use_local_llm": False
        }
        
        success, response = self.run_test(
            "AI Improve Regex",
            "POST",
            "ai-improve-regex",
            200,
            data=ai_request
        )
        return success

def main():
    print("🚀 Starting Tally Statement Processor API Tests")
    print("=" * 60)
    
    tester = TallyProcessorAPITester()
    
    # Run all tests in sequence
    tests = [
        tester.test_health_check,
        tester.test_create_client,
        tester.test_get_clients,
        tester.test_get_client_by_id,
        tester.test_file_upload,
        tester.test_confirm_mapping,
        tester.test_regex_patterns,
        tester.test_get_client_patterns,
        tester.test_ai_improve_regex
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed - check backend implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())