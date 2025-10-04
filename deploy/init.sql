-- ExoHunter Database Initialization Script
-- This script sets up the initial database schema and sample data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS exohunter;
CREATE SCHEMA IF NOT EXISTS logs;

-- Set search path
SET search_path = exohunter, public;

-- Create tables for predictions logging
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_type VARCHAR(50) NOT NULL,
    input_data JSONB NOT NULL,
    prediction_result JSONB NOT NULL,
    model_type VARCHAR(100),
    confidence_score FLOAT,
    processing_time_ms FLOAT,
    client_ip INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create table for API request logs
CREATE TABLE IF NOT EXISTS logs.api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    method VARCHAR(10) NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms FLOAT,
    request_size INTEGER,
    response_size INTEGER,
    client_ip INET,
    user_agent TEXT,
    request_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create table for model metadata
CREATE TABLE IF NOT EXISTS model_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL UNIQUE,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    feature_count INTEGER,
    classes JSONB,
    training_metrics JSONB,
    file_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction_type ON predictions(prediction_type);
CREATE INDEX IF NOT EXISTS idx_predictions_model_type ON predictions(model_type);
CREATE INDEX IF NOT EXISTS idx_api_requests_created_at ON logs.api_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_api_requests_path ON logs.api_requests USING gin(path gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_api_requests_status_code ON logs.api_requests(status_code);
CREATE INDEX IF NOT EXISTS idx_model_metadata_name ON model_metadata(model_name);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_predictions_updated_at BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_metadata_updated_at BEFORE UPDATE ON model_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample model metadata
INSERT INTO model_metadata (
    model_name, 
    model_version, 
    model_type, 
    feature_count, 
    classes,
    training_metrics
) VALUES (
    'random_forest_baseline',
    '1.0.0',
    'RandomForestClassifier',
    8,
    '["CANDIDATE", "FALSE POSITIVE", "CONFIRMED"]'::jsonb,
    '{"accuracy": 0.85, "precision": 0.83, "recall": 0.87, "f1_score": 0.85}'::jsonb
) ON CONFLICT (model_name) DO NOTHING;

-- Create views for analytics
CREATE OR REPLACE VIEW prediction_stats AS
SELECT 
    prediction_type,
    model_type,
    COUNT(*) as total_predictions,
    AVG(confidence_score) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    DATE_TRUNC('day', created_at) as prediction_date
FROM predictions
GROUP BY prediction_type, model_type, DATE_TRUNC('day', created_at)
ORDER BY prediction_date DESC;

CREATE OR REPLACE VIEW api_stats AS
SELECT 
    path,
    method,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count,
    DATE_TRUNC('hour', created_at) as request_hour
FROM logs.api_requests
GROUP BY path, method, DATE_TRUNC('hour', created_at)
ORDER BY request_hour DESC;

-- Grant permissions
GRANT USAGE ON SCHEMA exohunter TO exohunter;
GRANT USAGE ON SCHEMA logs TO exohunter;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA exohunter TO exohunter;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA logs TO exohunter;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA exohunter TO exohunter;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA logs TO exohunter;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA exohunter GRANT ALL ON TABLES TO exohunter;
ALTER DEFAULT PRIVILEGES IN SCHEMA logs GRANT ALL ON TABLES TO exohunter;
ALTER DEFAULT PRIVILEGES IN SCHEMA exohunter GRANT ALL ON SEQUENCES TO exohunter;
ALTER DEFAULT PRIVILEGES IN SCHEMA logs GRANT ALL ON SEQUENCES TO exohunter;

COMMIT;
