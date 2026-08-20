import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { AgentCoreStack } from '../lib/cdk-stack';

test('AgentCoreStack synthesizes a configured memory resource', () => {
  const app = new cdk.App();
  const stack = new AgentCoreStack(app, 'TestStack', {
    spec: {
      name: 'testproject',
      version: 1,
      managedBy: 'CDK' as const,
      runtimes: [],
      memories: [
        {
          name: 'TestMemory',
          eventExpiryDuration: 30,
          strategies: [
            {
              type: 'SEMANTIC',
              namespaces: ['/users/{actorId}/facts'],
            },
          ],
        },
      ],
      credentials: [],
      evaluators: [],
      onlineEvalConfigs: [],
      policyEngines: [],
      agentCoreGateways: [],
      mcpRuntimeTools: [],
      unassignedTargets: [],
    },
  });
  const template = Template.fromStack(stack);
  template.resourceCountIs('AWS::BedrockAgentCore::Memory', 1);
  template.hasOutput('StackNameOutput', {
    Description: 'Name of the CloudFormation Stack',
  });
});
