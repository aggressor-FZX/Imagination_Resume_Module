const fs = require('fs');
const AjvDefault = require('ajv');
const Ajv2020 = require('./node_modules/ajv/dist/2020');
const addFormats = require('ajv-formats');

const researchSchema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "taskIds": { "type": "string" },
    "filePaths": { "type": "string" },
    "customContext": { "type": "string" },
    "includeProjectTree": { "type": "boolean" },
    "detailLevel": { "type": "string", "enum": ["low", "medium", "high"] },
    "saveTo": { "type": "string" },
    "saveToFile": { "type": "boolean" },
    "projectRoot": { "type": "string" },
    "tag": { "type": "string" }
  },
  "required": ["query","projectRoot"],
  "additionalProperties": false
};

console.log('Trying Ajv default (require("ajv"))...');
try {
  const Ajv = AjvDefault;
  const ajv = new Ajv();
  addFormats(ajv);
  const validate = ajv.compile(researchSchema);
  console.log('Ajv default compiled schema OK');
} catch (err) {
  console.error('Ajv default failed:', err && err.message);
}

console.log('\nTrying Ajv draft-2020 entrypoint...');
try {
  const Ajv2020Ctor = Ajv2020;
  const ajv2020 = new Ajv2020Ctor();
  addFormats(ajv2020);
  const validate2020 = ajv2020.compile(researchSchema);
  console.log('Ajv draft-2020 compiled schema OK');
} catch (err) {
  console.error('Ajv draft-2020 failed:', err && err.message);
}
