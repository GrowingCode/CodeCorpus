package com.mingrisoft.DOM_demo;

import java.io.File;
import java.io.IOException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.xml.sax.SAXException;

/**
 * @author bwm
 * 
 */
public class BookDOMAdd {

	private Document document = null;

	private String path = null;

	public BookDOMAdd(String path) {
		// TODO Auto-generated constructor stub
		this.path = path;
	}

	/**
	 * 读取XML文件
	 * 
	 * @param path
	 * @return
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 */
	private void parseReadFile() {
		DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory
				.newInstance();
		DocumentBuilder dombuilder = null;
		try {
			dombuilder = documentBuilderFactory.newDocumentBuilder();
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		File file = new File(path);
		try {
			document = dombuilder.parse(file);
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/**
	 * 添加book子节点
	 */
	public void addBook() {
		parseReadFile();

		Element book = document.createElement("book:book");
		Element name = document.createElement("book:name");
		Element publisher = document.createElement("book:publisher");
		Element company = document.createElement("book:company");
		Element author = document.createElement("book:author");
		Element isbn = document.createElement("book:ISBN");
		Element price = document.createElement("book:price");
		Element url = document.createElement("book:url");

		name.setTextContent("《Java全能速查宝典》");
		publisher.setTextContent("人民邮电出版社");
		company.setTextContent("明日科技");
		author.setTextContent("梁冰");
		isbn.setTextContent("9787115214874");
		price.setTextContent("59.00");
		price.setAttribute("unit", "yuan");
		price.setAttribute("unitType", "RMB");
		url.setTextContent("http://www.mingribook.com/bookinfo.php?id=191");

		book.appendChild(name);
		book.appendChild(publisher);
		book.appendChild(company);
		book.appendChild(author);
		book.appendChild(isbn);
		book.appendChild(price);
		book.appendChild(url);

		document.getDocumentElement().appendChild(book);
	}

	/**
	 * 创建XML文件
	 * 
	 * @param node
	 *            DOM节点
	 * @param url
	 *            XML文件路径
	 */
	public void writeFile(String out) {
		TransformerFactory transformerFactory = TransformerFactory
				.newInstance();
		DOMSource domSource = new DOMSource(document);
		StreamResult streamResult = new StreamResult(new File(out));
		try {
			Transformer transformer = transformerFactory.newTransformer();
			transformer.transform(domSource, streamResult);
		} catch (TransformerConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (TransformerException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public static void main(String[] arg) {
		String in = "xmldemo/BookDOMBulid.xml";
		String out = "xmldemo/BookDOMAdd.xml";
		BookDOMAdd bulid = new BookDOMAdd(in);
		bulid.addBook();
		bulid.writeFile(out);
	}
}
