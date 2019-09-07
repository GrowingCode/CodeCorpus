/**
 * @jdk版本:1.6
 * @编码时间:2010-3-20
 */
package com.mingrisoft.entity;

import java.util.Map;

/**
 * @author bwm
 * 
 */
public class BookPrice {

	// 属性名称和属性值列表
	private Map<String, String> attributeMap;
	// 价格值
	private Double value;

	public Map<String, String> getAttributeMap() {
		return this.attributeMap;
	}

	public void setAttributeMap(Map<String, String> attributeMap) {
		this.attributeMap = attributeMap;
	}

	public Double getValue() {
		return this.value;
	}

	public void setValue(Double value) {
		this.value = value;
	}

	@Override
	public String toString() {
		StringBuffer buffer = new StringBuffer();
		buffer.append("[atrributes=" + this.attributeMap + "\n");
		buffer.append("vlaue=" + this.value + "\n]");
		return buffer.toString();
	}
}
