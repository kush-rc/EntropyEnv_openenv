# server/datasets/security_cases.py
# Ground truth cases for MCP Security Sandbox tasks.
# Graders compare agent answers against these cases.

SECURITY_CASES = {
    'sec_easy': [
        {
            'case_id': 'sec_easy_001',
            'task_subtype': 'single',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'sql_injection',
            'cvss_range': [7.5, 9.8],
            'expected_severity': 'critical',
            'required_fix_tokens': ['?', 'execute', 'param'],
            'must_preserve_identifier': 'execute_query',
            'current_feedback_keywords': ['sql', 'injection'],
            'original_vuln_pattern': "query + user_input",
            'tool_call': 'db.execute("SELECT * FROM users WHERE name = \'" + user_input + "\'")',
            'task_description': 'Identify the SQL injection vulnerability in this MCP database tool call and propose a fix.',
        },
        {
            'case_id': 'sec_easy_002',
            'task_subtype': 'single',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'xss',
            'cvss_range': [4.0, 6.5],
            'expected_severity': 'medium',
            'required_fix_tokens': ['escape', 'html'],
            'must_preserve_identifier': 'render_html',
            'current_feedback_keywords': ['xss', 'script'],
            'original_vuln_pattern': '<script>',
            'tool_call': 'render_html(f"<div>{user_comment}</div>")',
            'task_description': 'Scan this HTML rendering call for XSS vulnerability.',
        },
        {
            'case_id': 'sec_easy_003',
            'task_subtype': 'single',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'hardcoded_secret',
            'cvss_range': [5.0, 7.5],
            'expected_severity': 'high',
            'required_fix_tokens': ['env', 'os'],
            'must_preserve_identifier': 'connect_db',
            'current_feedback_keywords': ['hardcoded', 'password'],
            'original_vuln_pattern': 'password="admin123"',
            'tool_call': 'connect_db(host="prod-db.internal", password="admin123", user="root")',
            'task_description': 'Scan this database connection call for hardcoded credentials.',
        },
        {
            'case_id': 'sec_easy_004',
            'task_subtype': 'single',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'path_traversal',
            'cvss_range': [6.0, 8.5],
            'expected_severity': 'high',
            'required_fix_tokens': ['basename', 'path'],
            'must_preserve_identifier': 'read_file',
            'current_feedback_keywords': ['path', 'traversal', '..'],
            'original_vuln_pattern': '../../../etc/passwd',
            'tool_call': 'read_file(filepath=user_input)  # user sends "../../etc/passwd"',
            'task_description': 'Scan this file read call for path traversal vulnerability.',
        },
        {
            'case_id': 'sec_easy_005',
            'task_subtype': 'single',
            'completion_threshold': 0.80,
            'max_steps': 4,
            'done_conditions': {'min_actions': 1, 'required_sequence': ['identify_vulnerability']},
            'expected_vuln_type': 'missing_auth',
            'cvss_range': [7.0, 9.0],
            'expected_severity': 'critical',
            'required_fix_tokens': ['auth', 'token', 'verify'],
            'must_preserve_identifier': 'delete_user',
            'current_feedback_keywords': ['authentication', 'missing'],
            'original_vuln_pattern': 'no auth check',
            'tool_call': 'delete_user(user_id=request.params["id"])  # no authentication check',
            'task_description': 'Scan this admin action call for missing authentication.',
        },
    ],
    'sec_medium': [
        {
            'case_id': 'sec_medium_001',
            'task_subtype': 'multi',
            'completion_threshold': 0.75,
            'max_steps': 6,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'sql_injection',
            'cvss_range': [7.5, 9.8],
            'expected_severity': 'critical',
            'required_fix_tokens': ['?', 'execute', 'param'],
            'must_preserve_identifier': 'search_products',
            'current_feedback_keywords': ['sql', 'union', 'injection'],
            'original_vuln_pattern': 'query + ',
            'reviewer_feedback': 'Your fix prevents basic injection but UNION-based attacks are still possible. Use parameterized queries with bound parameters.',
            'additional_vulns': [
                {'vuln_type': 'xss', 'severity': 'medium', 'line': 15},
                {'vuln_type': 'missing_auth', 'severity': 'high', 'line': 3},
            ],
            'tool_call': '''def search_products(query, user_id):
    # No auth check on line 3
    results = db.execute("SELECT * FROM products WHERE name LIKE '%" + query + "%'")
    return f"<div>{results}</div>"  # XSS on line 15''',
            'task_description': 'This MCP tool has multiple security issues. Identify the most critical vulnerability, propose a fix, and revise based on reviewer feedback.',
        },
        {
            'case_id': 'sec_medium_002',
            'task_subtype': 'multi',
            'completion_threshold': 0.75,
            'max_steps': 6,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'ssrf',
            'cvss_range': [6.5, 9.0],
            'expected_severity': 'high',
            'required_fix_tokens': ['allowlist', 'url'],
            'must_preserve_identifier': 'fetch_url',
            'current_feedback_keywords': ['ssrf', 'internal', 'url'],
            'original_vuln_pattern': 'requests.get(user_url)',
            'reviewer_feedback': 'Fix addresses external URLs but internal network requests (169.254.x.x, localhost) are still allowed. Add an allowlist or block private IP ranges.',
            'additional_vulns': [
                {'vuln_type': 'hardcoded_secret', 'severity': 'medium', 'line': 2},
            ],
            'tool_call': '''def fetch_url(user_url):
    API_KEY = "sk-secret-key-12345"  # hardcoded secret
    resp = requests.get(user_url, headers={"Authorization": API_KEY})
    return resp.text''',
            'task_description': 'Scan this URL fetcher for all vulnerabilities including SSRF and credential exposure.',
        },
        {
            'case_id': 'sec_medium_003',
            'task_subtype': 'multi',
            'completion_threshold': 0.75,
            'max_steps': 6,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'idor',
            'cvss_range': [5.5, 8.0],
            'expected_severity': 'high',
            'required_fix_tokens': ['user_id', 'owner'],
            'must_preserve_identifier': 'get_document',
            'current_feedback_keywords': ['idor', 'authorization', 'object'],
            'original_vuln_pattern': 'doc_id from request',
            'reviewer_feedback': 'Fix checks user ownership but does not validate the requesting user session. Add session-based auth check.',
            'additional_vulns': [
                {'vuln_type': 'rate_limit_missing', 'severity': 'medium', 'line': 1},
            ],
            'tool_call': '''@app.get("/documents/{doc_id}")  # no rate limiting
def get_document(doc_id: int, request: Request):
    # No check if requesting user owns this document
    return db.documents.find_one({"_id": doc_id})''',
            'task_description': 'Scan this document access endpoint for authorization and rate limiting issues.',
        },
    ],
    'sec_hard': [
        {
            'case_id': 'sec_hard_001',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'sql_injection',
            'cvss_range': [8.0, 10.0],
            'expected_severity': 'critical',
            'required_fix_tokens': ['?', 'execute', 'param'],
            'must_preserve_identifier': 'send_email',
            'current_feedback_keywords': ['still vulnerable', 'sql still present', 'parameterize'],
            'original_vuln_pattern': "query + user_input",
            'reviewer_feedback': 'Fix still allows SQL injection through the search parameter. Use parameterized queries.',
            'reviewer_feedback_sequence': [
                'Fix still allows SQL injection through the search parameter. Use parameterized queries.',
                'Better, but concatenation on line 5 still vulnerable. Also add input validation.',
            ],
            'tool_call': '''def send_email(recipient, subject, body):
    query = "INSERT INTO emails (to_addr, subject, body) VALUES ('" + recipient + "', '" + subject + "', '" + body + "')"
    db.execute(query)
    log(f"Email sent to {recipient}")
    return {"status": "sent", "to": recipient}''',
            'task_description': 'Rewrite this email tool call to remove SQL injection while preserving ALL business logic. A reviewer will evaluate your fix.',
        },
        {
            'case_id': 'sec_hard_002',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'jwt_misuse',
            'cvss_range': [7.5, 9.5],
            'expected_severity': 'critical',
            'required_fix_tokens': ['verify', 'algorithm', 'HS256', 'secret'],
            'must_preserve_identifier': 'authenticate_user',
            'current_feedback_keywords': ['jwt', 'algorithm', 'none attack', 'verify signature'],
            'original_vuln_pattern': 'algorithms=["none"]',
            'reviewer_feedback': 'Fix does not prevent algorithm=none attack. Explicitly set allowed algorithms.',
            'reviewer_feedback_sequence': [
                'Fix does not prevent algorithm=none attack. Explicitly set allowed algorithms.',
                'Algorithm is set but secret key is still derived from user input. Use server secret.',
            ],
            'tool_call': '''def authenticate_user(token):
    payload = jwt.decode(token, options={"verify_signature": False})
    user_id = payload.get("user_id")
    return get_user(user_id)''',
            'task_description': 'Rewrite this JWT authentication to prevent algorithm confusion attacks while preserving user lookup logic.',
        },
        {
            'case_id': 'sec_hard_003',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'xxe',
            'cvss_range': [7.0, 9.5],
            'expected_severity': 'high',
            'required_fix_tokens': ['disable', 'external', 'entities'],
            'must_preserve_identifier': 'parse_xml',
            'current_feedback_keywords': ['xxe', 'entity', 'external'],
            'original_vuln_pattern': 'allow_external_entities=True',
            'reviewer_feedback': 'Fix disables DTD but doesn\'t disable external entities. Set both to False.',
            'reviewer_feedback_sequence': [
                'Fix disables DTD but doesn\'t disable external entities. Set both to False.',
                'Better, but you should also validate XML schema against a whitelist.',
            ],
            'tool_call': '''def parse_xml(xml_string):
    import xml.etree.ElementTree as ET
    parser = ET.XMLParser()  # Default parser allows XXE
    tree = ET.fromstring(xml_string, parser)
    return tree.findall('.//user/name')''',
            'task_description': 'Fix the XXE vulnerability in this XML parser while preserving all business logic. Reviewer will test your fix against attack payloads.',
        },
        {
            'case_id': 'sec_hard_004',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'ssrf',
            'cvss_range': [7.5, 9.0],
            'expected_severity': 'critical',
            'required_fix_tokens': ['allowlist', 'private', 'ip'],
            'must_preserve_identifier': 'fetch_webhook',
            'current_feedback_keywords': ['ssrf', 'internal', 'network'],
            'original_vuln_pattern': 'requests.get(url)',
            'reviewer_feedback': 'Fix blocks 127.0.0.1 but allows localhost and private IPs (10.x, 172.x, 192.168.x).',
            'reviewer_feedback_sequence': [
                'Fix blocks 127.0.0.1 but allows localhost and private IPs (10.x, 172.x, 192.168.x).',
                'Good, but you should also block IPv6 loopback (::1) and link-local addresses.',
            ],
            'tool_call': '''def fetch_webhook(url):
    import requests
    # Webhook URL from user input
    response = requests.get(url, timeout=10)
    return response.json()''',
            'task_description': 'Prevent SSRF attacks on this webhook fetcher. Reviewer will try internal network access bypasses.',
        },
        {
            'case_id': 'sec_hard_005',
            'task_subtype': 'adversarial',
            'completion_threshold': 0.70,
            'max_steps': 8,
            'done_conditions': {'min_actions': 3, 'required_sequence': ['identify_vulnerability', 'propose_fix', 'revise_fix']},
            'expected_vuln_type': 'idor',
            'cvss_range': [6.0, 8.5],
            'expected_severity': 'high',
            'required_fix_tokens': ['owner', 'session', 'user_id'],
            'must_preserve_identifier': 'update_profile',
            'current_feedback_keywords': ['idor', 'authorization', 'owner'],
            'original_vuln_pattern': 'profile_id from request',
            'reviewer_feedback': 'Fix checks profile ownership but uses user_id from request body (attacker-controlled).',
            'reviewer_feedback_sequence': [
                'Fix checks profile ownership but uses user_id from request body (attacker-controlled).',
                'Better, but session validation is weak. Use cryptographic session tokens, not just user_id in cookie.',
            ],
            'tool_call': '''@app.post("/profile/update")
def update_profile(profile_id: int, user_id: int, data: dict):
    # user_id comes from request body (!)
    profile = db.profiles.find_one({"_id": profile_id})
    profile.update(data)
    return {"status": "updated"}''',
            'task_description': 'Fix IDOR vulnerability allowing users to edit others\' profiles. Reviewer will test horizontal privilege escalation.',
        },
    ],
}
