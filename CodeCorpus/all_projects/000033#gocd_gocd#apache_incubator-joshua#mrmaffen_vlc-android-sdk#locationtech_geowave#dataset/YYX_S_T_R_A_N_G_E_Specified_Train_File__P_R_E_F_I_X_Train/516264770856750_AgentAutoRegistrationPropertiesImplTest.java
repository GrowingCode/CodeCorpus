/*
 * Copyright 2016 ThoughtWorks, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.thoughtworks.go.agent;

import com.thoughtworks.go.config.AgentAutoRegistrationProperties;
import org.apache.commons.io.FileUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

public class AgentAutoRegistrationPropertiesImplTest {

    @Rule
    public TemporaryFolder folder = new TemporaryFolder();
    private File configFile;

    @Before
    public void setUp() throws IOException {
        configFile = folder.newFile();
    }

    @Test
    public void shouldReturnAgentAutoRegisterPropertiesIfPresent() throws Exception {
        Properties properties = new Properties();

        properties.put(AgentAutoRegistrationPropertiesImpl.AGENT_AUTO_REGISTER_KEY, "foo");
        properties.put(AgentAutoRegistrationPropertiesImpl.AGENT_AUTO_REGISTER_RESOURCES, "foo, zoo");
        properties.put(AgentAutoRegistrationPropertiesImpl.AGENT_AUTO_REGISTER_ENVIRONMENTS, "foo, bar");
        properties.put(AgentAutoRegistrationPropertiesImpl.AGENT_AUTO_REGISTER_HOSTNAME, "agent01.example.com");
        properties.store(new FileOutputStream(configFile), "");

        AgentAutoRegistrationProperties reader = new AgentAutoRegistrationPropertiesImpl(configFile);
        assertThat(reader.agentAutoRegisterKey(), is("foo"));
        assertThat(reader.agentAutoRegisterResources(), is("foo, zoo"));
        assertThat(reader.agentAutoRegisterEnvironments(), is("foo, bar"));
        assertThat(reader.agentAutoRegisterHostname(), is("agent01.example.com"));
    }

    @Test
    public void shouldReturnEmptyStringIfPropertiesNotPresent() {
        AgentAutoRegistrationProperties reader = new AgentAutoRegistrationPropertiesImpl(configFile);
        assertThat(reader.agentAutoRegisterKey().isEmpty(), is(true));
        assertThat(reader.agentAutoRegisterResources().isEmpty(), is(true));
        assertThat(reader.agentAutoRegisterEnvironments().isEmpty(), is(true));
        assertThat(reader.agentAutoRegisterHostname().isEmpty(), is(true));
    }

    @Test
    public void shouldScrubTheAutoRegistrationProperties() throws Exception {
        String originalContents = "" +
                "#\n" +
                "# file autogenerated by chef, any changes will be lost\n" +
                "#\n" +
                "# the registration key\n" +
                "agent.auto.register.key = some secret key\n" +
                "\n" +
                "# the resources on this agent\n" +
                "agent.auto.register.resources = some,resources\n" +
                "\n" +
                "# The hostname of this agent\n" +
                "agent.auto.register.hostname = agent42.example.com\n" +
                "\n" +
                "# The environments this agent belongs to\n" +
                "agent.auto.register.environments = production,blue\n" +
                "\n";
        FileUtils.write(configFile, originalContents);

        AgentAutoRegistrationProperties properties = new AgentAutoRegistrationPropertiesImpl(configFile);
        properties.scrubRegistrationProperties();

        String newContents = "" +
                "#\n" +
                "# file autogenerated by chef, any changes will be lost\n" +
                "#\n" +
                "# the registration key\n" +
                "# The autoregister key has been intentionally removed by Go as a security measure.\n" +
                "# agent.auto.register.key = some secret key\n" +
                "\n" +
                "# the resources on this agent\n" +
                "# This property has been removed by Go after attempting to auto-register with the Go server.\n" +
                "# agent.auto.register.resources = some,resources\n" +
                "\n" +
                "# The hostname of this agent\n" +
                "# This property has been removed by Go after attempting to auto-register with the Go server.\n" +
                "# agent.auto.register.hostname = agent42.example.com\n" +
                "\n" +
                "# The environments this agent belongs to\n" +
                "# This property has been removed by Go after attempting to auto-register with the Go server.\n" +
                "# agent.auto.register.environments = production,blue\n" +
                "\n";
        assertThat(FileUtils.readFileToString(configFile), is(newContents));
    }
}
